

agents = [agent for agent.history_loc in U.base_model.agents]

 plt.hist2d(a[600,0::2],a[600,1::2],normed = False,bins = 20,range = np.array([[0,200]