    g = (g - np.mean(g))/np.std(g)
    f = (f - np.mean(f))/np.std(f)
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))
    padh = Hk // 2
    padw = Wk // 2
    img_pad = zero_pad(f, padh, padw)
    for i in range(Hi):
        for j in range(Wi):
            img_patch = img_pad[i:i+Hk, j:j+Wk]
            img_normalize = (img_patch - np.mean(img_patch)) / np.std(img_patch)
            out[i, j] = (img_normalize * g).sum()