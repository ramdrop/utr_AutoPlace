//////////////////////////////////////////////////////////////////////////////////////////////
/// \file mcransac.cpp
///
/// \author Keenan Burnett
/// \brief Rigid and motion-compensated RANSAC implementations along with some auxilliary
///     SE(3) math functions.
//////////////////////////////////////////////////////////////////////////////////////////////
#include <complex>
#include "mcransac.hpp"

void enforce_orthogonality(Eigen::MatrixXd &R) {
    if (R.cols() == 3) {
        const Eigen::Vector3d col1 = R.block(0, 1, 3, 1).normalized();
        const Eigen::Vector3d col2 = R.block(0, 2, 3, 1).normalized();
        const Eigen::Vector3d newcol0 = col1.cross(col2);
        const Eigen::Vector3d newcol1 = col2.cross(newcol0);
        R.block(0, 0, 3, 1) = newcol0;
        R.block(0, 1, 3, 1) = newcol1;
        R.block(0, 2, 3, 1) = col2;
    } else if (R.cols() == 2) {
        const double epsilon = 0.001;
        if (fabs(R(0, 0) - R(1, 1)) > epsilon || fabs(R(1, 0) + R(0, 1)) > epsilon) {
            std::cout << "ERROR: this is not a proper rigid transformation!" << std::endl;
        }
        double a = (R(0, 0) + R(1, 1)) / 2;
        double b = (-R(1, 0) + R(0, 1)) / 2;
        double sum = sqrt(pow(a, 2) + pow(b, 2));
        a /= sum;
        b /= sum;
        R(0, 0) = a; R(0, 1) = b;
        R(1, 0) = -b; R(1, 1) = a;
    }
}

void get_rigid_transform(Eigen::MatrixXd p1, Eigen::MatrixXd p2, Eigen::MatrixXd &Tf) {
    assert(p1.cols() == p2.cols() && p1.rows() == p2.rows());
    const int dim = p1.rows();
    Eigen::VectorXd mu1 = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd mu2 = mu1;
    // Calculate centroid of each point cloud
    for (int i = 0; i < p1.cols(); ++i) {
        mu1 += p1.block(0, i, dim, 1);
        mu2 += p2.block(0, i, dim, 1);
    }
    mu1 /= p1.cols();
    mu2 /= p1.cols();
    // Subtract centroid from each cloud
    Eigen::MatrixXd q1 = p1;
    Eigen::MatrixXd q2 = p2;
    for (int i = 0; i < p1.cols(); ++i) {
        q1.block(0, i, dim, 1) -= mu1;
        q2.block(0, i, dim, 1) -= mu2;
    }
    // Calculate rotation using SVD
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);
    for (int i = 0; i < p1.cols(); ++i) {
        H += q1.block(0, i, dim, 1) * q2.block(0, i, dim, 1).transpose();
    }
    // auto svd = H.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::MatrixXd R_hat = V * U.transpose();
    if (R_hat.determinant() < 0) {
        V.block(0, dim - 1, dim, 1) = -1 * V.block(0, dim - 1, dim, 1);
        R_hat = V * U.transpose();
    }
    if (R_hat.determinant() != 1.0)
        enforce_orthogonality(R_hat);
    // Calculate translation
    Eigen::VectorXd t = mu2 - R_hat * mu1;
    // Create the output transformation
    Tf = Eigen::MatrixXd::Identity(dim + 1, dim + 1);
    Tf.block(0, 0, dim, dim) = R_hat;
    Tf.block(0, dim, dim, 1) = t;
}

std::vector<int> random_subset(int max_index, int subset_size) {
    std::vector<int> subset;
    if (max_index < 0 || subset_size < 0)
        return subset;
    if (max_index < subset_size)
        subset_size = max_index;
    subset = std::vector<int>(subset_size, -1);
    for (uint i = 0; i < subset.size(); i++) {
        while (subset[i] < 0) {
            int idx = std::rand() % max_index;
            if (std::find(subset.begin(), subset.begin() + i, idx) == subset.begin() + i)
                subset[i] = idx;
        }
    }
    return subset;
}

Eigen::MatrixXd carrot(Eigen::VectorXd x) {
    Eigen::MatrixXd X;
    assert(x.rows() == 3 || x.rows() == 6);
    if (x.rows() == 3) {
        X = Eigen::MatrixXd::Zero(3, 3);
        X << 0, -x(2), x(1),
             x(2), 0, -x(0),
             -x(1), x(0), 0;
    } else {
        X = Eigen::MatrixXd::Zero(4, 4);
        X << 0, -x(5), x(4), x(0),
             x(5), 0, -x(3), x(1),
             -x(4), x(3), 0, x(2),
             0, 0, 0, 1;
    }
    return X;
}

// x: 4 x 1, output: 4 x 6
Eigen::MatrixXd circledot(Eigen::VectorXd x) {
    assert(x.rows() == 4);
    Eigen::Vector3d rho = x.block(0, 0, 3, 1);
    double eta = x(3);
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(4, 6);
    X.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity() * eta;
    X.block(0, 3, 3, 3) = -1 * carrot(rho);
    return X;
}

// Lie Vector xi = [rho, phi]^T (6 x 1) --> SE(3) T = [C, R; 0 0 0 1] (4 x 4)
Eigen::Matrix4d se3ToSE3(Eigen::MatrixXd xi) {
    assert(xi.rows() == 6);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    Eigen::Vector3d rho = xi.block(0, 0, 3, 1);
    Eigen::Vector3d phibar = xi.block(3, 0, 3, 1);
    double phi = phibar.norm();
    Eigen::MatrixXd C = Eigen::MatrixXd::Identity(3, 3);
    if (phi != 0) {
        phibar.normalize();
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        C = cos(phi) * I + (1 - cos(phi)) * phibar * phibar.transpose() + sin(phi) * carrot(phibar);
        enforce_orthogonality(C);
        Eigen::Matrix3d J = I * sin(phi) / phi + (1 - sin(phi) / phi) * phibar * phibar.transpose() +
            carrot(phibar) * (1 - cos(phi)) / phi;
        rho = J * rho;
    }
    T.block(0, 0, 3, 3) = C;
    T.block(0, 3, 3, 1) = rho;
    return T;
}

// SE(3) T = [C, R; 0 0 0 1] (4 x 4) --> Lie Vector xi = [rho, phi]^T (6 x 1)
Eigen::VectorXd SE3tose3(Eigen::MatrixXd T) {
    Eigen::MatrixXd C = T.block(0, 0, 3, 3);
    Eigen::MatrixXcd Cc = C.cast<std::complex<double>>();
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces;
    ces.compute(Cc);
    int idx = -1;
    Eigen::VectorXcd evalues = ces.eigenvalues();
    Eigen::MatrixXcd evectors = ces.eigenvectors();
    for (int i = 0; i < 3; ++i) {
        if (evalues(i, 0).real() != 0 && evalues(i, 0).imag() == 0) {
            idx = i;
            break;
        }
    }
    assert(idx != -1);
    Eigen::VectorXd abar = Eigen::Vector3d::Zero();
    for (int i = 0; i < abar.rows(); ++i) {
        abar(i, 0) = evectors(i, idx).real();
    }
    abar.normalize();
    double trace = 0;
    for (int i = 0; i < C.rows(); ++i) {
        trace += C(i, i);
    }
    double phi = acos((trace - 1) / 2);
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d J = I * sin(phi) / phi + (1 - sin(phi) / phi) * abar * abar.transpose() +
        carrot(abar) * (1 - cos(phi)) / phi;
    Eigen::VectorXd rho = J.inverse() * T.block(0, 3, 3, 1);
    Eigen::VectorXd xi = Eigen::VectorXd::Zero(6);
    xi.block(0, 0, 3, 1) = rho;
    xi.block(3, 0, 3, 1) = phi * abar;
    return xi;
}

double wrapto2pi(double theta) {
    if (theta < 0) {
        return theta + 2 * M_PI;
    } else if (theta >= 2 * M_PI) {
        return theta - 2 * M_PI;
    } else {
        return theta;
    }
}

double Ransac::computeModel() {
    uint max_inliers = 0;
    std::vector<int> best_inliers;
    int dim = p1.rows();
    int subset_size = 2;
    int i = 0;
    for (i = 0; i < iterations; ++i) {
        std::vector<int> subset = random_subset(p1.cols(), subset_size);
        if ((int)subset.size() < subset_size)
            continue;
        // Compute transform from the random sample
        Eigen::MatrixXd p1small, p2small;
        p1small = Eigen::MatrixXd::Zero(dim, subset_size);
        p2small = p1small;
        for (int j = 0; j < subset_size; ++j) {
            p1small.block(0, j, dim, 1) = p1.block(0, subset[j], dim, 1);
            p2small.block(0, j, dim, 1) = p2.block(0, subset[j], dim, 1);
        }
        Eigen::MatrixXd T_current;
        get_rigid_transform(p1small, p2small, T_current);
        // Check the number of inliers
        std::vector<int> inliers;
        getInliers(T_current, inliers);
        if (inliers.size() > max_inliers) {
            best_inliers = inliers;
            max_inliers = inliers.size();
        }
        if (double(inliers.size()) / double(p1.cols()) > inlier_ratio)
            break;
    }
    // Refine transformation using the inlier set
    Eigen::MatrixXd p1small, p2small;
    p1small = Eigen::MatrixXd::Zero(dim, best_inliers.size());
    p2small = p1small;
    for (uint j = 0; j < best_inliers.size(); ++j) {
        p1small.block(0, j, dim, 1) = p1.block(0, best_inliers[j], dim, 1);
        p2small.block(0, j, dim, 1) = p2.block(0, best_inliers[j], dim, 1);
    }
    get_rigid_transform(p1small, p2small, T_best);
    return double(max_inliers) / double(p1.cols());
}

void Ransac::getInliers(Eigen::MatrixXd Tf, std::vector<int> &inliers) {
    int dim = p1.rows();
    Eigen::MatrixXd p1_prime = Eigen::MatrixXd::Ones(dim + 1, p1.cols());
    p1_prime.block(0, 0, dim, p1.cols()) = p1;
    p1_prime = Tf * p1_prime;
    inliers.clear();
    for (uint i = 0; i < p1_prime.cols(); ++i) {
        auto distance = (p1_prime.block(0, i, dim, 1) - p2.block(0, i, dim, 1)).norm();
        if (distance < tolerance)
            inliers.push_back(i);
    }
}

void MCRansac::dopplerCorrection(Eigen::VectorXd wbar, Eigen::VectorXd &p) {
    double v = fabs(wbar(0, 0));
    double rsq = p(0) * p(0) + p(1) * p(1);
    p(0) += beta * v * p(0) * p(0) / rsq;
    p(1) += beta * v * p(0) * p(1) / rsq;
}

void MCRansac::get_motion_parameters(std::vector<int> subset, Eigen::VectorXd &wbar) {
    double lastError = 0;
    for (int it = 0; it < max_gn_iterations; ++it) {
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6, 6);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(6);
        for (uint m = 0; m < subset.size(); ++m) {
            Eigen::MatrixXd Tbar = se3ToSE3(delta_ts[subset[m]] * wbar);
            Eigen::VectorXd p1 = p1bar.col(subset[m]);
            Eigen::VectorXd p2 = p2bar.col(subset[m]);
            if (doppler) {
                dopplerCorrection(wbar, p1);
                dopplerCorrection(wbar, p2);
            }
            Eigen::VectorXd gbar = Tbar * p1;
            Eigen::MatrixXd G = delta_ts[subset[m]] * circledot(gbar);
            Eigen::VectorXd ebar = p2 - gbar;
            A += G.transpose() * G;
            b += G.transpose() * ebar;
        }
        // Eigen::VectorXd delta_w = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
        Eigen::VectorXd delta_w = A.ldlt().solve(b);
        // Line search for best update
        double minError = 10000000;
        double bestAlpha = 1.0;
        for (double alpha = 0.1; alpha <= 1.0; alpha += 0.1) {
            double e = 0;
            Eigen::VectorXd wbar_temp = wbar + alpha * delta_w;
            for (uint m = 0; m < subset.size(); ++m) {
                Eigen::VectorXd p1 = p1bar.col(subset[m]);
                Eigen::VectorXd p2 = p2bar.col(subset[m]);
                if (doppler) {
                    dopplerCorrection(wbar, p1);
                    dopplerCorrection(wbar, p2);
                }
                Eigen::MatrixXd Tbar = se3ToSE3(delta_ts[subset[m]] * wbar_temp);
                Eigen::VectorXd ebar = p2 - Tbar * p1;
                e += ebar.squaredNorm();
            }
            if (e < minError) {
                minError = e;
                bestAlpha = alpha;
            }
        }
        wbar = wbar + bestAlpha * delta_w;
        if (delta_w.squaredNorm() < epsilon_converge)
            break;
        if (it > 0 && fabs((lastError - minError) / lastError) < error_converge)
            break;
        lastError = minError;
    }
}

static int get_closest(double x, std::vector<double> v) {
    int closest = 0;
    double distance = 1000;
    for (uint i = 0; i < v.size(); ++i) {
        double d = fabs(x - v[i]);
        if (d < distance) {
            closest = i;
            distance = d;
        }
    }
    return closest;
}

void MCRansac::getInliers(Eigen::VectorXd wbar, std::vector<int> &inliers) {
    // Use a discrete number of transforms to speed this up
    std::vector<Eigen::MatrixXd> transforms(num_transforms);
    for (int i = 0; i < num_transforms; ++i) {
        transforms[i] = se3ToSE3(delta_vec[i] * wbar);
    }
    for (uint i = 0; i < p1bar.cols(); ++i) {
        Eigen::VectorXd p2 = p2bar.block(0, i, 4, 1);
        Eigen::VectorXd p1 = p1bar.block(0, i, 4, 1);
        if (doppler) {
            dopplerCorrection(wbar, p1);
            dopplerCorrection(wbar, p2);
        }
        Eigen::VectorXd error = p2 - transforms[get_closest(delta_ts[i], delta_vec)] * p1;
        if (error.squaredNorm() < tolerance)
            inliers.push_back(i);
    }
}

int MCRansac::getNumInliers(Eigen::VectorXd wbar) {
    // Use a discrete number of transforms to speed this up
    std::vector<Eigen::MatrixXd> transforms(num_transforms);
    for (int i = 0; i < num_transforms; ++i) {
        transforms[i] = se3ToSE3(delta_vec[i] * wbar);
    }
    int inliers = 0;
    for (uint i = 0; i < p1bar.cols(); ++i) {
        Eigen::VectorXd p2 = p2bar.block(0, i, 4, 1);
        Eigen::VectorXd p1 = p1bar.block(0, i, 4, 1);
        if (doppler) {
            dopplerCorrection(wbar, p1);
            dopplerCorrection(wbar, p2);
        }
        Eigen::VectorXd error = p2 - transforms[get_closest(delta_ts[i], delta_vec)] * p1;
        if (error.squaredNorm() < tolerance)
            inliers++;
    }
    return inliers;
}

void MCRansac::getTransform(double delta_t, Eigen::MatrixXd &Tf) {
    Tf = se3ToSE3(w_best * delta_t);
}

double MCRansac::computeModel() {
    int max_inliers = 0;
    int subset_size = 2;
    int i = 0;
    for (i = 0; i < iterations; ++i) {
        std::vector<int> subset = random_subset(p1bar.cols(), subset_size);
        Eigen::VectorXd wbar = Eigen::VectorXd::Zero(6, 1);
        get_motion_parameters(subset, wbar);
        int inliers = getNumInliers(wbar);
        if (inliers > max_inliers) {
            max_inliers = inliers;
            w_best = wbar;
        }
        if (double(inliers) / double(p1bar.cols()) > inlier_ratio)
            break;
    }
    // Refine transformation using the inlier set
    std::vector<int> best_inliers;
    getInliers(w_best, best_inliers);
    get_motion_parameters(best_inliers, w_best);
    return double(best_inliers.size()) / double(p1bar.cols());
}

static int get_closest(std::vector<float> vec, float value) {
    int closest = 0;
    float mind = 10000;
    for (uint i = 0; i < vec.size(); ++i) {
        float d = fabs(vec[i] - value);
        if (d < mind) {
            mind = d;
            closest = i;
        }
    }
    return closest;
}

Eigen::Matrix4d get_inverse_tf(Eigen::Matrix4d T) {
    Eigen::Matrix3d R = T.block(0, 0, 3, 3);
    Eigen::Matrix4d T2 = Eigen::Matrix4d::Identity();
    T2.block(0, 0, 3, 3) = R.transpose();
    T2.block(0, 3, 3, 1) = -1 * R.transpose() * T.block(0, 3, 3, 1);
    return T2;
}
