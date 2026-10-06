import numpy as np

from heritage_eval.pipeline import apply_mask, fleiss_kappa_binary, ssim_Lab


def test_mask_shape_and_content():
    img = np.ones((20, 20, 3), dtype=np.uint8) * 255
    _, mask = apply_mask(img, [(2, 2), (17, 2), (17, 17), (2, 17)])
    assert mask.shape == (20, 20)
    assert mask[10, 10] == 255
    assert mask[0, 0] == 0


def test_identical_images_have_high_ssim():
    img = np.ones((64, 64, 3), dtype=np.uint8) * 120
    mask = np.ones((64, 64), dtype=np.uint8) * 255
    assert ssim_Lab(img, img, mask) > 0.999


def test_fleiss_kappa_perfect_agreement():
    # Three raters: all select item 1 and reject item 2.
    R = np.array([[1, 0], [1, 0], [1, 0]], dtype=int)
    kappa, observed, _ = fleiss_kappa_binary(R)
    assert abs(kappa - 1.0) < 1e-6
    assert abs(observed - 1.0) < 1e-6
