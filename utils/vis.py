import io
import PIL.Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from torchvision.transforms import ToTensor
from utils.utils import enforce_orthog, get_inverse_tf, get_T_ba
from utils.utils import getApproxTimeStamps, wrapto2pi

def convert_plt_to_img():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return PIL.Image.open(buf)

def convert_plt_to_tensor():
    return ToTensor()(convert_plt_to_img())

def draw_batch(batch, out, config):
    """Creates an image of the radar scan, scores, and keypoint matches for a single batch."""
    # Draw radar image
    radar = batch['data'][0].squeeze().numpy()
    plt.subplots()
    plt.imshow(radar, cmap='gray')
    radar_img = convert_plt_to_tensor()

    # Draw keypoint matches
    src = out['src'][0].squeeze().detach().cpu().numpy()
    tgt = out['tgt'][0].squeeze().detach().cpu().numpy()
    match_weights = out['match_weights'][0].squeeze().detach().cpu().numpy()

    nms = config['vis_keypoint_nms']
    max_w = np.max(match_weights)
    plt.imshow(radar, cmap='gray')
    for i in range(src.shape[0]):
        if match_weights[i] < nms * max_w:
            continue
        plt.plot([src[i, 0], tgt[i, 0]], [src[i, 1], tgt[i, 1]], c='w', linewidth=2, zorder=2)
        plt.scatter(src[i, 0], src[i, 1], c='g', s=5, zorder=3)
        plt.scatter(tgt[i, 0], tgt[i, 1], c='r', s=5, zorder=4)
    match_img = convert_plt_to_tensor()

    # Draw scores
    scores = out['scores'][0].squeeze().detach().cpu().numpy()
    plt.subplots()
    plt.imshow(scores, cmap='inferno')
    score_img = convert_plt_to_tensor()

    return vutils.make_grid([radar_img, score_img, match_img])

def draw_matches(batch, out, config, solver):
    azimuth_step = (2 * np.pi) / 400
    cart_pixel_width = config['cart_pixel_width']
    cart_resolution = config['cart_resolution']
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    T_met_pix = np.array([[0, -cart_resolution, 0, cart_min_range],
                          [cart_resolution, 0, 0, -cart_min_range],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
    T_pix_met = np.linalg.inv(T_met_pix)

    keypoint_ints = out['keypoint_ints']
    ids = torch.nonzero(keypoint_ints[0, 0] > 0, as_tuple=False).squeeze(1)
    src = out['src_rc'][0, ids].squeeze().detach().cpu().numpy()
    tgt = out['tgt_rc'][0, ids].squeeze().detach().cpu().numpy()
    radar = batch['data'][0].squeeze().numpy()
    _, axs = plt.subplots(1, 3, tight_layout=True)
    # Raw locations overlayed, no transforms
    axs[0].imshow(radar, cmap='gray', extent=(0, 640, 640, 0), interpolation='none')
    axs[0].set_axis_off()
    axs[0].set_title('raw')
    for i in range(src.shape[0]):
        axs[0].plot([src[i, 0], tgt[i, 0]], [src[i, 1], tgt[i, 1]], c='w', linewidth=1, zorder=2)
        axs[0].scatter(src[i, 0], src[i, 1], c='limegreen', s=2, zorder=3)
        axs[0].scatter(tgt[i, 0], tgt[i, 1], c='r', s=2, zorder=4)

    src = out['src'][0, ids].squeeze().detach().cpu().numpy()
    tgt = out['tgt'][0, ids].squeeze().detach().cpu().numpy()
    # Use Rigid Transform
    axs[1].imshow(radar, cmap='gray', extent=(0, 640, 640, 0), interpolation='none')
    axs[1].set_axis_off()
    axs[1].set_title('rigid')
    T_tgt_src = get_T_ba(out, a=0, b=1)
    error = np.zeros((src.shape[0], 2))
    for i in range(src.shape[0]):
        x1 = np.array([src[i, 0], src[i, 1], 0, 1]).reshape(4, 1)
        x2 = np.array([tgt[i, 0], tgt[i, 1], 0, 1]).reshape(4, 1)
        x1 = T_tgt_src @ x1
        e = x1 - x2
        error[i, 1] = np.sqrt(e.T @ e)
        error[i, 0] = int(wrapto2pi(np.arctan2(x2[1, 0], x2[0, 0])) // azimuth_step)
        x1 = T_pix_met @ x1
        x2 = T_pix_met @ x2
        axs[1].plot([x1[0, 0], x2[0, 0]], [x1[1, 0], x2[1, 0]], c='w', linewidth=1, zorder=2)
        axs[1].scatter(x1[0, 0], x1[1, 0], c='limegreen', s=2, zorder=3)
        axs[1].scatter(x2[0, 0], x2[1, 0], c='r', s=2, zorder=4)

    # Use Interpolated Poses
    t1 = batch['timestamps'][0].numpy().squeeze()
    t2 = batch['timestamps'][1].numpy().squeeze()
    times1 = getApproxTimeStamps([src], [t1])[0]
    times2 = getApproxTimeStamps([tgt], [t2])[0]
    t_refs = batch['t_ref'].numpy()

    T_1a = np.identity(4, dtype=np.float32)
    T_1b = np.identity(4, dtype=np.float32)
    axs[2].imshow(radar, cmap='gray', extent=(0, 640, 640, 0), interpolation='none')
    axs[2].set_axis_off()
    axs[2].set_title('interp')
    error2 = np.zeros((src.shape[0], 2))
    for i in range(src.shape[0]):
        solver.getPoseBetweenTimes(T_1a, times1[i], t_refs[1, 0, 0])
        solver.getPoseBetweenTimes(T_1b, times2[i], t_refs[1, 0, 0])
        x1 = np.array([src[i, 0], src[i, 1], 0, 1]).reshape(4, 1)
        x2 = np.array([tgt[i, 0], tgt[i, 1], 0, 1]).reshape(4, 1)
        x1 = T_1a @ x1
        x2 = T_1b @ x2
        e = x1 - x2
        error2[i, 1] = np.sqrt(e.T @ e)
        error2[i, 0] = int(wrapto2pi(np.arctan2(x2[1, 0], x2[0, 0])) // azimuth_step)
        x1 = T_pix_met @ x1
        x2 = T_pix_met @ x2
        axs[2].plot([x1[0, 0], x2[0, 0]], [x1[1, 0], x2[1, 0]], c='w', linewidth=1, zorder=2)
        axs[2].scatter(x1[0, 0], x1[1, 0], c='limegreen', s=2, zorder=3)
        axs[2].scatter(x2[0, 0], x2[1, 0], c='r', s=2, zorder=4)

    plt.savefig('matches.pdf', bbox_inches='tight', pad_inches=0.0)
    plt.figure()

    idx = np.argsort(error[:, 0])
    error = error[idx, :]
    idx = np.argsort(error2[:, 0])
    error2 = error2[idx, :]
    plt.plot(error[:, 0], error[:, 1], color='b', label='raw error', linewidth=1)
    plt.plot(error2[:, 0], error2[:, 1], color='r', label='interp error', linewidth=1)
    plt.title('raw error')
    plt.legend()
    plt.savefig('matches2.pdf', bbox_inches='tight', pad_inches=0.0)

def draw_batch_steam(batch, out, config):
    """Creates an image of the radar scan, scores, and keypoint matches for a single batch."""
    # Draw radar image
    radar = batch['data'][0].squeeze().numpy()
    radar_tgt = batch['data'][-1].squeeze().numpy()
    plt.imshow(np.concatenate((radar, radar_tgt), axis=1), cmap='gray')
    plt.title('radar src-tgt pair')
    radar_img = convert_plt_to_tensor()

    # Draw keypoint matches
    src = out['src_rc'][-1].squeeze().detach().cpu().numpy()
    tgt = out['tgt_rc'][-1].squeeze().detach().cpu().numpy()
    keypoint_ints = out['keypoint_ints']

    ids = torch.nonzero(keypoint_ints[-1, 0] > 0, as_tuple=False).squeeze(1)
    ids_cpu = ids.cpu()

    plt.imshow(np.concatenate((radar, radar_tgt), axis=1), cmap='gray')
    delta = radar.shape[1]
    for i in range(src.shape[0]):
        if i in ids_cpu:
            custom_colour = 'g'
            plt.plot([src[i, 0], tgt[i, 0] + delta], [src[i, 1], tgt[i, 1]], c='y', linewidth=0.5, zorder=2)
            plt.scatter(src[i, 0], src[i, 1], c=custom_colour, s=5, zorder=3)
            plt.scatter(tgt[i, 0] + delta, tgt[i, 1], c=custom_colour, s=5, zorder=4)
    plt.title('matches')
    match_img = convert_plt_to_tensor()

    plt.imshow(np.concatenate((radar, radar_tgt), axis=0), cmap='gray')
    delta = radar.shape[1]
    for i in range(src.shape[0]):
        if i in ids_cpu:
            custom_colour = 'g'
            plt.plot([src[i, 0], tgt[i, 0]], [src[i, 1], tgt[i, 1] + delta], c='y', linewidth=0.5, zorder=2)
            plt.scatter(src[i, 0], src[i, 1], c=custom_colour, s=5, zorder=3)
            plt.scatter(tgt[i, 0], tgt[i, 1] + delta, c=custom_colour, s=5, zorder=4)
    plt.title('matches')
    match_img2 = convert_plt_to_tensor()

    # Draw scores
    scores = out['scores'][-1]
    if scores.size(0) == 3:
        scores = scores[1] + scores[2]
    scores = scores.squeeze().detach().cpu().numpy()
    plt.imshow(scores, cmap='inferno')
    plt.colorbar()
    plt.title('log det weight (weight score vis)')
    score_img = convert_plt_to_tensor()

    # Draw detector scores
    detector_scores = out['detector_scores'][-1].squeeze().detach().cpu().numpy()
    plt.imshow(detector_scores, cmap='inferno')
    plt.colorbar()
    plt.title('detector score')
    dscore_img = convert_plt_to_tensor()

    # Draw point-to-point error
    src_p = out['src'][-1].squeeze().T
    tgt_p = out['tgt'][-1].squeeze().T
    R_tgt_src = out['R'][0, -1, :2, :2]
    t_st_in_t = out['t'][0, -1, :2, :]
    error = tgt_p - (R_tgt_src @ src_p + t_st_in_t)
    error2_sqrt = torch.sqrt(torch.sum(error * error, dim=0).squeeze())
    error2_sqrt = error2_sqrt[ids_cpu].detach().cpu().numpy()

    plt.imshow(radar, cmap='gray')
    plt.scatter(src[ids_cpu, 0], src[ids_cpu, 1], c=error2_sqrt, s=5, zorder=2, cmap='rainbow')
    plt.clim(0.0, 1)
    plt.colorbar()
    plt.title('P2P error')
    p2p_img = convert_plt_to_tensor()

    return vutils.make_grid([dscore_img, score_img, radar_img]), vutils.make_grid([match_img, match_img2]), \
        vutils.make_grid([p2p_img])

def plot_sequences(T_gt, T_pred, seq_lens, returnTensor=True, T_icra=None, savePDF=False, fnames=None, flip=True):
    """Creates a top-down plot of the predicted odometry results vs. ground truth."""
    seq_indices = []
    idx = 0
    for s in seq_lens:
        seq_indices.append(list(range(idx, idx + s - 1)))
        idx += (s - 1)

    matplotlib.rcParams.update({'font.size': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
                                'axes.linewidth': 1.5, 'font.family': 'serif', 'pdf.fonttype': 42})
    T_flip = np.identity(4)
    T_flip[1, 1] = -1
    T_flip[2, 2] = -1
    imgs = []
    for seq_i, indices in enumerate(seq_indices):
        T_gt_ = np.identity(4)
        T_pred_ = np.identity(4)
        T_icra_ = np.identity(4)
        if flip:
            T_gt_ = np.matmul(T_flip, T_gt_)
            T_pred_ = np.matmul(T_flip, T_pred_)
        x_gt = []
        y_gt = []
        x_pred = []
        y_pred = []
        x_icra = []
        y_icra = []
        for i in indices:
            T_gt_ = np.matmul(T_gt[i], T_gt_)
            T_pred_ = np.matmul(T_pred[i], T_pred_)
            enforce_orthog(T_gt_)
            enforce_orthog(T_pred_)
            T_gt_temp = get_inverse_tf(T_gt_)
            T_pred_temp = get_inverse_tf(T_pred_)
            x_gt.append(T_gt_temp[0, 3])
            y_gt.append(T_gt_temp[1, 3])
            x_pred.append(T_pred_temp[0, 3])
            y_pred.append(T_pred_temp[1, 3])
            if T_icra is not None:
                T_icra_ = np.matmul(T_icra[i], T_icra_)
                enforce_orthog(T_icra_)
                T_icra_temp = get_inverse_tf(T_icra_)
                x_icra.append(T_icra_temp[0, 3])
                y_icra.append(T_icra_temp[1, 3])

        plt.figure(figsize=(10, 10), tight_layout=True)
        plt.grid(color='k', which='both', linestyle='--', alpha=0.75, dashes=(8.5, 8.5))
        plt.axes().set_aspect('equal')
        plt.plot(x_gt, y_gt, 'k', linewidth=2.5, label='GT')
        if x_icra and y_icra:
            plt.plot(x_icra, y_icra, 'r', linewidth=2.5, label='MC-RANSAC')
        plt.plot(x_pred, y_pred, 'b', linewidth=2.5, label='HERO')
        plt.xlabel('x (m)', fontsize=16)
        plt.ylabel('y (m)', fontsize=16)
        plt.legend(loc="upper left", edgecolor='k', fancybox=False)
        if savePDF and fnames is not None:
            plt.savefig(fnames[seq_i], bbox_inches='tight', pad_inches=0.0)
        if returnTensor:
            imgs.append(convert_plt_to_tensor())
        else:
            imgs.append(convert_plt_to_img())
    return imgs
