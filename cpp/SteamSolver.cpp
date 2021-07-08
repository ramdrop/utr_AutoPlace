//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SteamSolver.cpp
///
/// \author David Yoon, Keenan Burnett
/// \brief A C++ class with a boost::python wrapper for optimizing odometry poses over a sliding
///     window with a motion prior.
//////////////////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include "SteamSolver.hpp"
#include "P2P3ErrorEval.hpp"
#include "SE2VelPriorEval.hpp"
#include "mcransac.hpp"

// Reset trajectory to identity poses and zero velocities
void SteamSolver::resetTraj() {
    Eigen::Matrix<double, 4, 4> eig_identity = Eigen::Matrix<double, 4, 4>::Identity();
    lgmath::se3::Transformation T_identity(eig_identity);
    Eigen::Matrix<double, 6, 1> zero_vel;
    zero_vel.setZero();
    states_.clear();
    for (uint k = 0; k < window_size_; ++k) {
        TrajStateVar temp;
        temp.time = steam::Time(k * dt_);
        temp.pose = steam::se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(T_identity));
        temp.velocity = steam::VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(zero_vel));
        states_.push_back(temp);
    }
}

// Slide window and initialize newest frame with constant velocity
void SteamSolver::slideTraj() {
    // drop first frame
    states_.pop_front();
    // set first frame to identity
    lgmath::se3::Transformation T_i0 = states_[0].pose->getValue().inverse();
    for (uint k = 0; k < states_.size(); ++k){
        lgmath::se3::Transformation T_ki = states_[k].pose->getValue();
        states_[k].pose->setValue(T_ki*T_i0);
    }
    // add new frame to end
    lgmath::se3::Transformation T_km1_i = states_.back().pose->getValue();
    Eigen::Matrix<double, 6, 1> xi = dt_ * states_.back().velocity->getValue();

    TrajStateVar temp;
    temp.time = states_.back().time + steam::Time(dt_);
    temp.pose = steam::se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(
        lgmath::se3::Transformation(xi)*T_km1_i));
    temp.velocity = steam::VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(
        states_.back().velocity->getValue()));
    states_.push_back(temp);
}

// Set the Qc inverse matrix with the diagonal of Qc
void SteamSolver::setQcInv(const np::ndarray& Qc_diag) {
    Eigen::Matrix<double, 6, 1> temp = numpyToEigen2D(Qc_diag);
    Qc_inv_.setZero();
    Qc_inv_.diagonal() = 1.0/temp.array();
}

// Set measurements
void SteamSolver::setMeas(const p::object& p2_list, const p::object& p1_list, const p::object& weight_list,
    const p::object& t2_list, const p::object& t1_list, const p::object& t_refs) {
    p2_ = toStdVector<np::ndarray>(p2_list);
    p1_ = toStdVector<np::ndarray>(p1_list);
    w_ = toStdVector<np::ndarray>(weight_list);
    t2_ = toStdVector<np::ndarray>(t2_list);
    t1_ = toStdVector<np::ndarray>(t1_list);
    t_refs_ = toStdVector<int64_t>(t_refs);
}

// Set extrinsic transform Tsv
void SteamSolver::setExtrinsicTsv(const np::ndarray& T_sv) {
    Eigen::Matrix4d T_sv_eig = numpyToEigen2D(T_sv);
    lgmath::se3::Transformation T_sv_lg(T_sv_eig);
    T_sv_ = steam::se3::FixedTransformEvaluator::MakeShared(T_sv_lg);
}

// Run optimization
void SteamSolver::optimize() {
    // Motion prior
    bool allowExtrapolation = true;
    traj = steam::se3::SteamTrajInterface(Qc_inv_, allowExtrapolation);
    traj_init = true;
    int64_t t0 = t_refs_[0];
    for (uint i = 0; i < states_.size(); ++i) {
        int64_t delta = t_refs_[i] - t_refs_[0];
        double delta_t = double(delta) / 1.0e6;
        states_[i].time = steam::Time(delta_t);
        TrajStateVar& state = states_.at(i);
        steam::se3::TransformStateEvaluator::Ptr temp = steam::se3::TransformStateEvaluator::MakeShared(state.pose);
        traj.add(state.time, temp, state.velocity);
        if (i == 0) {  // lock first pose
            state.pose->setLock(true);
        }
    }
    // Cost Terms
    steam::ParallelizedCostTermCollection::Ptr costTerms(new steam::ParallelizedCostTermCollection());
    if (vel_prior_) {
        Eigen::Matrix<double, 6, 1> velocity = states_[0].velocity->getValue();
        // velocity[1] = 0;  // encourage lateral velocity to be close to zero
        Eigen::Matrix<double, 6, 6> vel_cov;
        vel_cov.setZero();
           Eigen::Array<double, 1, 6> vel_cov_diag;
        vel_cov_diag << 1, 1, 1, 1, 1, 1;
        vel_cov.diagonal() = vel_cov_diag;
        traj.addVelocityPrior(steam::Time(0.0), velocity, vel_cov);
    }
    traj.appendPriorCostTerms(costTerms);

    steam::L2LossFunc::Ptr sharedLossFuncL2(new steam::L2LossFunc());
    steam::GemanMcClureLossFunc::Ptr sharedLossFuncGM(new steam::GemanMcClureLossFunc(1.0));
    // loop through every frame
    for (uint i = 1; i < window_size_; ++i) {
        steam::se3::TransformStateEvaluator::Ptr T_k0_eval_ptr =
            steam::se3::TransformStateEvaluator::MakeShared(states_[i].pose);

        std::vector<int> inliers;
        if (use_ransac) {
            srand(t0 / 1e6);  // fix random seed for repeatability
            Eigen::VectorXd motion_vec = Eigen::VectorXd::Zero(6);
            Eigen::MatrixXd T;
            if (ransac_version == 1) {
                MCRansac mcransac(p1_[i-1], p2_[i-1], t1_[i-1], t2_[i-1]);
                mcransac.computeModel();
                mcransac.getMotion(motion_vec);
                mcransac.getInliers(motion_vec, inliers);
                mcransac.getTransform(0.25, T);
            } else {
                Ransac ransac(p1_[i-1], p2_[i-1]);
                ransac.computeModel();
                ransac.getTransform(T);
                ransac.getInliers(T, inliers);
            }
        } else {
            for (uint j = 0; j < p1_[i-1].shape(0); ++j) {
                inliers.push_back(j);
            }
        }
        // Only run STEAM on inliers from MCRANSAC (if use_ransac == true)
        for (uint k = 0; k < inliers.size(); ++k) {
            uint j = inliers[k];
            Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
            for (uint r = 0; r < 3; ++r) {
                for (uint c = 0; c < 3; ++c) {
                    R(r, c) = p::extract<float>(w_[i - 1][j][r][c]);
                }
            }
            steam::BaseNoiseModel<3>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<3>(R, steam::INFORMATION));
            // get measurement
            Eigen::Vector4d read;
            read << double(p::extract<float>(p2_[i-1][j][0])), double(p::extract<float>(p2_[i-1][j][1])),
                double(p::extract<float>(p2_[i-1][j][2])), 1.0;

            Eigen::Vector4d ref;
            ref << double(p::extract<float>(p1_[i-1][j][0])), double(p::extract<float>(p1_[i-1][j][1])),
                double(p::extract<float>(p1_[i-1][j][2])), 1.0;

            steam::se3::TransformEvaluator::Ptr T_eval_ptr;
            if (ct_steam) {
                int64_t ta_ = int64_t(p::extract<int64_t>(t1_[i-1][j])) - t0;
                int64_t tb_ = int64_t(p::extract<int64_t>(t2_[i-1][j])) - t0;
                double ta = double(ta_) / 1.0e6;
                double tb = double(tb_) / 1.0e6;
                steam::se3::TransformEvaluator::ConstPtr Ta0 = traj.getInterpPoseEval(steam::Time(ta));
                steam::se3::TransformEvaluator::ConstPtr Tb0 = traj.getInterpPoseEval(steam::Time(tb));
                T_eval_ptr = steam::se3::composeInverse(
                    steam::se3::compose(T_sv_, Tb0),
                    steam::se3::compose(T_sv_, Ta0));  // Tba = Tb0 * inv(Ta0)
            } else {
                T_eval_ptr = steam::se3::composeInverse(
                    steam::se3::compose(T_sv_, T_k0_eval_ptr), T_sv_);
            }
            steam::P2P3ErrorEval::Ptr error(new steam::P2P3ErrorEval(ref, read, T_eval_ptr));
            steam::WeightedLeastSqCostTerm<3, 6>::Ptr cost(
                new steam::WeightedLeastSqCostTerm<3, 6>(error, sharedNoiseModel, sharedLossFuncGM));
            costTerms->add(cost);
        }
    }
    if (zero_vel_prior_flag_) {
        Eigen::Matrix<double, 3, 3> vel_prior_noise = 1e-3 * Eigen::Matrix<double, 3, 3>::Identity();
        steam::BaseNoiseModel<3>::Ptr vel_prior_noise_model(new steam::StaticNoiseModel<3>(vel_prior_noise));
        for (uint i = 0; i < states_.size(); ++i) {
            steam::SE2VelPriorEval::Ptr error(new steam::SE2VelPriorEval(states_[i].velocity));
            steam::WeightedLeastSqCostTerm<3, 6>::Ptr cost(
                new steam::WeightedLeastSqCostTerm<3, 6>(error, vel_prior_noise_model, sharedLossFuncL2));
            costTerms->add(cost);
        }  // end i
    }
    steam::OptimizationProblem problem;
    // Add state variables
    for (uint i = 0; i < states_.size(); ++i) {
        const TrajStateVar& state = states_.at(i);
        problem.addStateVariable(state.pose);
        problem.addStateVariable(state.velocity);
    }
    problem.addCostTerm(costTerms);
    SolverType::Params params;
    params.verbose = false;
    solver_ = SolverBasePtr(new SolverType(&problem, params));
    solver_->optimize();
}

void SteamSolver::getPoses(np::ndarray& poses) {
    for (uint i = 0; i < states_.size(); ++i) {
        Eigen::Matrix<double, 4, 4> Tsi =
            T_sv_->evaluate().matrix()*states_[i].pose->getValue().matrix()*T_sv_->evaluate().inverse().matrix();
        for (uint r = 0; r < 3; ++r) {
            for (uint c = 0; c < 4; ++c) {
                poses[i][r][c] = float(Tsi(r, c));
            }
        }
    }
}

// Replaces pose (4x4) with T_ba
void SteamSolver::getPoseBetweenTimes(np::ndarray& pose, const int64_t ta, const int64_t tb) {
    if (!traj_init) {
        std::cout << "WARNING: traj not yet initialized" << std::endl;
        return;
    }
    int64_t t0 = t_refs_[0];
    int64_t ta_ = int64_t(ta) - t0;
    int64_t tb_ = int64_t(tb) - t0;
    steam::se3::TransformEvaluator::ConstPtr Ta0 = traj.getInterpPoseEval(steam::Time(double(ta_) / 1.0e6));
    steam::se3::TransformEvaluator::ConstPtr Tb0 = traj.getInterpPoseEval(steam::Time(double(tb_) / 1.0e6));
    steam::se3::TransformEvaluator::Ptr T_eval_ptr = steam::se3::composeInverse(
        steam::se3::compose(T_sv_, Tb0),
        steam::se3::compose(T_sv_, Ta0));  // Tba = Tb0 * inv(Ta0)
    Eigen::Matrix<double, 4, 4> Ta0_s = T_eval_ptr->evaluate().matrix();
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) {
            pose[i][j] = float(Ta0_s(i, j));
        }
    }
}

void SteamSolver::getVelocities(np::ndarray& vels) {
    for (uint i = 0; i < states_.size(); ++i) {
        Eigen::Matrix<double, 6, 1> vel = states_[i].velocity->getValue();
        for (uint r = 0; r < 6; ++r) {
            vels[i][r] = float(vel(r));
        }
    }
}

void SteamSolver::getSigmapoints2N(np::ndarray& sigma_T) {
    // query covariance at once
    std::vector<steam::StateKey> keys;
    keys.reserve(window_size_ - 1);
    for (unsigned int i = 1; i < states_.size(); i++) {
        // skip i = 0 since it's always locked
        const TrajStateVar& state = states_.at(i);
        keys.push_back(state.pose->getKey());
    }
    steam::BlockMatrix cov_blocks = solver_->queryCovarianceBlock(keys);
    // useful constants
    int n = 6;  // pose is 6D
    double alpha = sqrt(double(n));
    // loop through every frame (skipping first since it's locked)
    for (unsigned int i = 1; i < window_size_; i++) {
        // mean pose
        const TrajStateVar& state = states_.at(i);
        Eigen::Matrix4d T_i0_eigen = state.pose->getValue().matrix();
        // get cov and LLT decomposition
        Eigen::Matrix<double, 6, 6> cov = cov_blocks.at(i - 1, i - 1);
        Eigen::LLT<Eigen::MatrixXd> lltcov(cov);
        Eigen::MatrixXd L = lltcov.matrixL();
        // sigmapoints
        for (int a = 0; a < n; ++a) {
            // delta for pose
            Eigen::Matrix4d T_sp = lgmath::se3::vec2tran(L.col(a).head<6>()*alpha);
            Eigen::Matrix4d T_sp_inv = lgmath::se3::vec2tran(-L.col(a).head<6>()*alpha);
            // positive/negative sigmapoints
            T_sp = T_sv_->evaluate().matrix()*T_sp*T_i0_eigen*T_sv_->evaluate().inverse().matrix();
            T_sp_inv = T_sv_->evaluate().matrix()*T_sp_inv*T_i0_eigen*T_sv_->evaluate().inverse().matrix();
            // set output
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    sigma_T[i-1][a][r][c] = float(T_sp(r, c));
                    sigma_T[i-1][a+n][r][c] = float(T_sp_inv(r, c));
                }
            }
        }
    }
}
