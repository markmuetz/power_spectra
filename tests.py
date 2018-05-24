"""Not currently used but may be useful in future"""

def tests(w):
    print('All 1')
    w_test = np.zeros_like(w[0, LEVEL_HEIGHT_INDEX].data)
    w_test[:, :] = 1
    import ipdb; ipdb.set_trace()
    calc_w_fft(w_test)
    input('Enter to continue')

    print('Centre points')
    w_test = np.zeros_like(w[0, LEVEL_HEIGHT_INDEX].data)
    w_test[255:257, 255:257] = 1
    calc_w_fft(w_test)
    input('Enter to continue')

    print('Band in x-dir')
    w_test = np.zeros_like(w[0, LEVEL_HEIGHT_INDEX].data)
    w_test[250:260, :] = 1
    calc_w_fft(w_test)
    input('Enter to continue')

    print('Centre circle r=10')
    w_test = np.zeros_like(w[0, LEVEL_HEIGHT_INDEX].data)
    y, x = np.indices((w_test.shape))
    # import ipdb; ipdb.set_trace()
    w_test[(x - 256)**2 + (y - 256)**2 < 100] = 1
    calc_w_fft(w_test)
    input('Enter to continue')

    print('Some sines/cosines in x/y.')
    w_test = np.zeros_like(w[0, LEVEL_HEIGHT_INDEX].data)
    y, x = np.indices((w_test.shape))
    sin_x_modes = [3, 7, 11, 17]
    cos_y_modes = [4, 6, 11, 22]
    for sin_x_mode in sin_x_modes:
        w_test += np.sin(sin_x_mode * 2 * np.pi * x / 512)
    for cos_y_mode in cos_y_modes:
        w_test += np.cos(cos_y_mode * 2 * np.pi * y / 512)
    calc_w_fft(w_test)
    input('Enter to continue')

    print('sinc')
    w_test = np.zeros_like(w[0, LEVEL_HEIGHT_INDEX].data)
    y, x = np.indices((w_test.shape))
    # 0.0001 stops div by zero errors.
    r = np.sqrt((x - 256.0001)**2 + (y - 256.0001)**2)
    w_test = np.sin(r) / r
    calc_w_fft(w_test)
    input('Enter to continue')



