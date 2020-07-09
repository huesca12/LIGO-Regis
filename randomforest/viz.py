from sklearn import tree
import matplotlib.pyplot as plt

def treeViz(X, y, clf):
    fn=list(X.columns.values)
    cn=list(y.unique())
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (100,25), dpi=300)
    tree.plot_tree(clf,
                   feature_names = fn,
                   class_names = cn,
                   filled = True,
                   fontsize = 10,
                   label = "all",
                   impurity = False);
    fig.savefig('tree.png')
