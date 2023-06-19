import mne 


builtin_montages = mne.channels.get_builtin_montages(descriptions=True)

in_10_10 = mne.channels.make_standard_montage("standard_1005")

print(in_10_10)


in_10_10.plot()  # 4D
fig = in_10_10.plot(kind="3d", show=False)  # 3D

fig = fig.gca().view_init(azim=70, elev=15)  # set view angle for tutorial
