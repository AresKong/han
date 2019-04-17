import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms

def visualization(document, attention_s, attention_w, **kw):
    ax = plt.gca()
    t = ax.transData
    canvas = ax.figure.canvas
    # Get some pastel shades for the colors
    n_color_s = 5
    n_color_w = 5
    color_s = plt.cm.BuPu(np.linspace(0, 0.5, n_color_s))
    color_w = plt.cm.Reds(np.linspace(0, 0.5, n_color_w))

    value_bins = np.linspace(np.array(attention_s).min(), np.array(attention_s).max(), n_color_s - 1)
    where = np.digitize(attention_s, value_bins)
    sent_colors = color_s[where]

    offset = 0
    for i in range(len(document)):
        word_bins = np.linspace(np.array(attention_w[i]).min(), np.array(attention_w[i]).max(), n_color_w - 1)
        word_index = np.digitize(attention_w[i], word_bins)
        word_colors = color_w[word_index]
        y = 0.9 - i*0.08

        for j in range(-1, len(document[i])):
            if j == -1:
                text = ax.text(0.05, y, attention_s[i], color='black', transform=t,
                               bbox=dict(facecolor=sent_colors[i], edgecolor=sent_colors[i]), **kw)
            else:
                text = ax.text(0.05, y, document[i][j], color='black',transform=t,
                    bbox=dict(facecolor=word_colors[j], edgecolor=word_colors[j]), **kw)
            text.draw(canvas.get_renderer())
            ex = text.get_window_extent()
            offset += ex.width+7
            if j == len(document[i])-1:
                width = ex.width+7-offset
                offset = 0
            else:
                width = ex.width+7
            t = transforms.offset_copy(
                text.get_transform(), x=width, units='dots')
    plt.show()

if __name__ == '__main__':
    
    control_st = [["all","the","action","."],
                   ["mother","is","drying","dishes","and","the","tap","water","is","overflowing","the","sink","and","running","one","the","floor","."],
                   ["and", "johnny", "'s", "trying", "to", "get", "some", "cookies", "."],
                   ["and", "his", "step", "stool", "is", "falling", "."],
                   ["and", "little", "girl", "is", "reaching", "her", "hand", "up", "for", "a", "cookie"],
                   ["and", "putting", "her", "hand", "to", "her", "mouth", "."],
                   ["oh", "oh", "well", ",", "that", "'s", "what", "'s", "happening", "."],
                   ["i", "do", "n't", "know", "."],
                   ["mother", "'s", "stepping", "i", "mean", "the", "lady", "'s", "stepping", "in", "the", "water", "that", "spilled", "on", "the", "floor", "."],
                   ["that", "'s", "all", "i", "can", "be", "sure", "of", "."]]
    
    control_s = np.array([0.2090,
                            0.1773,
                            0.2520,
                            0.1346,
                            0.0804,
                            0.0804,
                            0.0761,
                            0.0357,
                            0.0224,
                            0.0097])

    control_w = np.array([[2.6955e-02,7.1212e-03,9.6561e-01,2.7614e-04],
                            [3.8871e-02,2.2386e-02,7.9671e-03,1.6109e-03,9.1974e-04,1.0695e-03,1.3551e-02,1.4354e-02,6.2803e-02,6.5284e-01,5.9725e-03,3.3820e-02,7.3512e-03,1.2009e-01,1.4393e-02,1.0059e-03,8.8871e-04,8.1505e-05],
                            [0.0038,0.0115,0.0659,0.7503,0.0637,0.0523,0.0285,0.0208,0.0019],
                            [0.0044,0.0097,0.1283,0.5899,0.1812,0.0814,0.0040],
                            [4.9602e-04,2.7493e-03,2.3126e-02,9.4323e-02,1.4708e-01,1.5496e-01,7.3834e-02,9.9893e-03,5.6160e-02,3.1114e-02,1.4337e-02],
                            [8.9631e-03,4.2688e-02,1.2581e-01,1.3746e-01,2.3855e-02,5.1264e-02,1.5470e-03,1.6445e-04],
                            [0.0111,0.0254,0.0327,0.0277,0.0334,0.0479,0.1187,0.1068,0.5868,0.0076],
                            [0.2155,0.2089,0.1668,0.3508,0.0388],
                            [1.2315e-01,3.5081e-02,1.6088e-01,3.5164e-02,2.1220e-02,8.4787e-03,9.5579e-03,3.3948e-02,4.1160e-01,4.2390e-02,6.0872e-03,2.1921e-02,1.0995e-02,3.0544e-02,3.8859e-02,4.5601e-03,4.8890e-03,5.0645e-04],
                            [1.3100e-03,6.0515e-03,6.8376e-02,8.2388e-02,2.2405e-01,3.9746e-01,1.0224e-01,1.1667e-01,1.1645e-03]
                            ])

    ad_st = [["well", "the", "girl", "is", "telling", "the", "boy", "to", "get", "the", "cookies", "down", "but", "do", "n't", "tell", "your", "mother", "."],
             ["and", "the", "boy", "is", "also", "falling", "over", "off", "the", "stool", "."],
             ["and", "the", "mother", "is", "letting", "the", "water", "run", "out", "of", "the", "sink", "."],
             ["and", "she", "'s", "drying", "dishes", "."],
             ["i", "do", "n't", "quite", "get", "that", "but", "then", "..."],
             ["uh", "she", "has", "water", "on", "the", "floor", "and", "and", "basically", "it", "'s", "kindof", "uh", "a", "distressing", "scene", "."],
             ["everything", "'s", "going", "haywire", "."],
             ["she", "needs", "to", "turn", "off", "the", "water", "."],
             ["if", "she", "turned", "off", "the", "water", "she", "'d", "be", "a", "hundred", "percent", "better", "off", "."]
             ]

    ad_s = np.array([0.1371,
                     0.1794,
                     0.1187,
                     0.1784,
                     0.0910,
                     0.1479,
                     0.0887,
                     0.0361,
                     0.0192])

    ad_w = np.array([[5.3468e-03,4.2870e-03,2.5486e-02,4.0991e-02,5.3225e-02,6.5311e-03,1.9870e-02,8.8534e-03,6.4515e-03,3.8749e-03,1.7734e-02,4.9346e-01,3.0047e-02,7.3696e-03,6.2710e-03,3.7921e-02,6.1142e-02,1.7058e-01,4.9565e-04],
             [9.8077e-04,3.8174e-03,3.9716e-02,1.5804e-01,2.8474e-01,1.3052e-01,4.0889e-02,3.8428e-02,1.8359e-02,2.8242e-01,1.7393e-03],
             [4.5269e-04,2.2183e-03,6.0730e-01,1.3670e-01,9.6523e-02,8.6798e-03,1.9588e-02,1.1134e-02,3.1058e-02,7.2105e-02,4.1231e-03,9.6315e-03,3.8886e-04],
             [0.0229,0.1079,0.3220,0.4870,0.0444,0.0092],
             [0.0384,0.0459,0.0640,0.2542,0.1317,0.1006,0.1258,0.2239,0.0127],
             [0.0039,0.0089,0.0182,0.0560,0.0634,0.0132,0.0309,0.0155,0.0172,0.0654,0.0478,0.1019,0.2380,0.1215,0.0692,0.1073,0.0197,0.0015],
             [0.0654,0.0970,0.5547,0.2733,0.0071],
             [0.0093,0.1785,0.1147,0.5759,0.0741,0.0178,0.0274,0.0016],
             [2.5411e-03,6.0617e-03,3.6957e-02,2.2391e-02,1.1691e-02,3.7521e-02,3.9264e-02,2.0040e-01,2.4746e-01,8.0631e-02,1.0109e-01,1.8816e-01,1.5831e-02,9.4978e-03,3.4468e-04]
             ])

    c_st = [["a", "girl", "'s", "reaching", "for", "a", "cookie", "."],
        ["the", "boy", "'s", "taking", "a", "cookie", "out", "of", "the", "cookie", "jar", "."],
        ["the", "bench", "is", "tumbling", "."],
        ["the", "sink", "is", "running", "over", "water", "."],
        ["the", "mother", "'s", "wiping", "the", "plate", "."],
        ["she", "'s", "also", "looking", "out", "the", "window", "."],
        ["and", "she", "'s", "standing", "in", "water", "."],
        ["and", "that", "'s", "it", "."]]

    c_w = np.array([[0.0971,0.0881,0.0665,0.2858,0.3441,0.1021,0.0146,0.0010],
                    [0.0018,0.0211,0.0311,0.1723,0.1185,0.0412,0.1731,0.3806,0.0285,0.0171,0.0128,0.0015],
                    [0.0078,0.2563,0.4459,0.2797,0.0083],
                    [9.1129e-04,3.7518e-02,1.3635e-01,8.0975e-01,1.1086e-02,4.0253e-03,2.3459e-04],
                    [1.8687e-03,8.1036e-01,6.5145e-02,5.2705e-02,8.2529e-03,6.0006e-02,1.3370e-03],
                    [0.0056,0.0256,0.3908,0.1431,0.2450,0.0418,0.1435,0.0037],
                    [1.4984e-04,9.2802e-04,7.4144e-03,9.7586e-01,1.4482e-02,1.0668e-03,6.6553e-05],
                    [0.0559,0.1035,0.2075,0.5565,0.0555]
    ])

    c_s = np.array([0.1998
                     ,0.2232
                     ,0.1228
                     ,0.1972
                     ,0.0964
                     ,0.0891
                     ,0.0599
                     ,0.0088])
    visualization(control_st, control_s, control_w, size=10)

    visualization(c_st, c_s, c_w, size=10)
