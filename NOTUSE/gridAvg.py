def gridAvg(pm,pn,xRes,yRes):

    import numpy as np
    
    '''Only works for constant xRes and yRes for now!'''
    
    ### Compute a spatially averaged field

    ## Append grid location information to the data metafile

    lon_g = np.arange(0,360,xRes)
    lat_g = np.arange(-90,90,yRes)

    # Define grid box centers
    lon_c = lon_g+xRes/2
    lat_c = lat_g+yRes/2

    # Define a new metadata file that has grid coordinates for this resolution choice
    pmg = pm;

    pmg.loc[:,'lat_ind'] = np.nan
    pmg.loc[:,'lon_ind'] = np.nan

    ## Determine lat_ind and lon_ind for every record

    # List of proxy sites
    # site_list = list(pn.columns.values)

    for index, row in pmg.iterrows():
        lon_s = row['Lon_E']
        lat_s = row['Lat_N']

        if lon_s<0:
            lon_s = 360+lon_s

        # Return the indices of the bins to which each value in input array belongs.
        lat_ind = np.digitize(lat_s,lat_g,right=True)
        lon_ind = np.digitize(lon_s,lon_g,right=True)
        pmg.set_value(index,'lat_ind',lat_ind-1)
        pmg.set_value(index,'lon_ind',lon_ind-1)

    ## Spatial averaging

    LT,LR = pn.shape
    # initialize gridded average field
    G = np.nan*np.ones([LT,len(lat_g),len(lon_g)])
    # Initialize field of number of obs
    Gn = np.nan*np.ones([len(lat_g),len(lon_g)])

    loi = pmg['lon_ind'].astype('int')
    lai = pmg['lat_ind'].astype('int')

    # Loop over grid lat
    for ii in range(0,len(lat_g)):
        # Loop over grid lon
        for jj in range(0,len(lon_g)):
            # find locations in this grid box
            # average them and store in G
            # store the number of obs in Gn
            Gn[ii,jj] =     len(pmg[(loi==jj) & (lai==ii)].index)
            if Gn[ii,jj] > 0:
                G[:,ii,jj] = pn[pmg[(loi==jj) & (lai==ii)].index].mean(1).values
    return G, pmg, lat_g,lon_g, Gn
