from matplotlib import rcParams

def configure_matplotlib():
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22

    rcParams.update({'figure.autolayout': True, 
                     'text.usetex': True,
                     'text.latex.preamble':r"\usepackage{amsmath}",
                     'legend.loc':'upper right',
                     'font.size': SMALL_SIZE,
                     'axes.titlesize': BIGGER_SIZE,
                     'axes.labelsize': MEDIUM_SIZE,
                     'xtick.labelsize': SMALL_SIZE,
                     'legend.fontsize': SMALL_SIZE,
                     'figure.titlesize': BIGGER_SIZE
                     })
    