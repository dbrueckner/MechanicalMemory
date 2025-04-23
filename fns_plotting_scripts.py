#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:22:39 2020

@author: D.Brueckner
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import dill as pickle
import os


def snippets_states(X,state,N_snippets_max,L,N_snip_per_traj_max=50):
    
    N_part = X.shape[0]
    N_t_max = X.shape[1]

    X_snippets = np.zeros((N_snippets_max,N_t_max,2))
    X_snippets[:] = np.nan
    
    X_snippet_indices = np.zeros((N_snippets_max))
    X_snippet_indices[:] = np.nan

    X_states = np.zeros((N_part,N_snip_per_traj_max))
    X_states[:] = np.nan
    
    X_succ = np.zeros((N_part,N_snip_per_traj_max))
    X_succ[:] = np.nan

    cnt=0
    for j in range(0,N_part):
        
        N_t = len([x for x in X[j,:] if not np.isnan(x)])
        
        array12_j = np.zeros(N_t)
        for t in range(0,N_t):
            if np.abs(X[j,t]) < L/2:
                array12_j[t] = 1 #bridge
            elif np.abs(X[j,t]) > L/2:
                array12_j[t] = 2 #island
            else:
                array12_j[t] = np.nan

        diffcellpos = np.diff(array12_j)
        N_transitions = len(np.where(diffcellpos != 0)[0])
        if N_transitions > 1:
            jumplocations = np.where(diffcellpos != 0)[0]+1
            jumplocations = np.concatenate((np.array([0]), jumplocations, np.array([N_t-1])))
            N_jumps = len(jumplocations)

            cnt_states=0
            for it in range(0,N_jumps-1):
                snippet = X[j,jumplocations[it]:jumplocations[it+1]]
                snippet_state = state[j,jumplocations[it]:jumplocations[it+1]]
                
                if array12_j[jumplocations[it]]==1 and len(snippet)>5:
                    fac = -np.sign(snippet[0])
                    X_snippets[cnt,:len(snippet),0] = fac*snippet
                    X_snippets[cnt,:len(snippet),1] = snippet_state
                    
                    X_snippet_indices[cnt] = j
                    
                    l = list(snippet_state)
                    majority_state = max(set(l), key = l.count)
                    
                    #mean_state = np.nanmean(snippet_state)
                    X_states[j,cnt_states] = majority_state#int(np.round(mean_state))
                    
                    if jumplocations[it]==0:
                        X_succ[j,cnt_states] = 0
                    elif jumplocations[it+1]==N_t-1:
                        X_succ[j,cnt_states] = 0
                    else:
                        if np.sign(X[j,jumplocations[it]-1]) != np.sign(X[j,jumplocations[it+1]+1]):
                            X_succ[j,cnt_states] = 1
                        else:
                            X_succ[j,cnt_states] = 0

                    cnt+=1
                    cnt_states+=1
    
    N_snip = len([x for x in X_snippets[:,0,1] if not np.isnan(x)])
    X_snippets = X_snippets[:N_snip,:,:]
    X_snippet_indices = X_snippet_indices[:N_snip]
    
    X_snippets_sorted = np.zeros((N_snip,N_t_max,2))
    X_snippets_sorted[:] = np.nan
    counts_states = np.zeros(2)
    for j in range(0,N_snip):
        N_t_here = len([x for x in X_snippets[j,:,1] if not np.isnan(x)])
        state_here = list(X_snippets[j,:N_t_here,1])
        majority_state = int(max(set(state_here), key = state_here.count))

        X_snippets_sorted[int(counts_states[majority_state]),:,majority_state] = X_snippets[j,:,0]
        counts_states[majority_state] += 1

    return X_snippets,X_states,X_snippet_indices,X_succ,X_snippets_sorted


def corr_vacf(X,delta_t,frac_part=1): #X[j,t]
    
    Ntrajectories = X.shape[0]
    N_t_max = X.shape[1]
    
    sum_tot = np.zeros(N_t_max) #only calculate up to N_t_max to save time
    CF12 = np.zeros(N_t_max)
    CF12_all = np.zeros((Ntrajectories,N_t_max))

    if(len(X.shape)==2):
        N_dim = 1
    elif(len(X.shape)==3):
        N_dim = X.shape[2]
        
    for j in range(0,Ntrajectories):
        if(N_dim>1):
            tracklength = (~np.isnan(X[0,:,0])).cumsum(0).argmax(0)+1
        elif(N_dim==1):
            tracklength = (~np.isnan(X[0,:])).cumsum(0).argmax(0)+1
            
        if(tracklength<N_t_max):
            delta_max = tracklength
        else:
            delta_max = N_t_max
        
        CF_j = np.zeros(N_t_max)
        for delta in range(0,delta_max):
            N_tot = 0
            N = tracklength - delta
            sum_CFj = 0
            for t in range(0,N):
                if(N_dim>1):
                    for d in range(0,N_dim):
                        if(~np.isnan(X[j,t+delta,d]) and ~np.isnan(X[j,t,d])):
                            sum_CFj += X[j,t+delta,d]*X[j,t,d]
                            if d==0:
                                N_tot += 1
                elif(N_dim==1):
                    if(~np.isnan(X[j,t+delta]) and ~np.isnan(X[j,t])):
                        sum_CFj += X[j,t+delta]*X[j,t]
                        N_tot += 1
            
            if N_tot>0:
                CF_j[delta] = sum_CFj/N_tot
        
        sum_tot += CF_j
        CF12_all[j,:] = CF_j
    
    for t in range(0,N_t_max):
        if(N_dim>1):
            Ntrajectories_t = int(sum(~np.isnan(X[:,t,0])))
        elif(N_dim==1):
            Ntrajectories_t = int(sum(~np.isnan(X[:,t])))
        
        if(Ntrajectories_t>0 and sum_tot[t] != 0): #also check for sum not being zero, as in principle also need trajs at t+delta
            CF12[t] = sum_tot[t]/Ntrajectories_t    
        else:
            CF12[t] = np.nan    
    
    time = np.linspace(0,N_t_max-1,N_t_max)*delta_t
    
    return CF12,CF12_all,time



def calc_probs(X_states):
    
    N_part = X_states.shape[0]
    N = np.sum(~np.isnan(X_states))
    prob_first = np.zeros(2)
    for state in [0,1]:
        prob_first[state] = np.sum(X_states==state)/N
    
    prob_second_sum = np.zeros((2,2))
    for state1 in [0,1]:
        for state2 in [0,1]:
            for j in range(0,N_part):
                N_states = np.sum(~np.isnan(X_states[j,:]))
                if N_states>1:
                    for it in range(1,N_states):
                        if X_states[j,it-1]==state1 and X_states[j,it]==state2:
            
                            prob_second_sum[state1,state2] += 1
    
    prob_second = np.zeros((2,2))
    for state1 in [0,1]:
        prob_second[state1,:] = prob_second_sum[state1,:]/np.sum(prob_second_sum[state1,:])
    
    prob_third_sum = np.zeros((2,2,2))
    for state1 in [0,1]:
        for state2 in [0,1]:
            for state3 in [0,1]:
                for j in range(0,N_part):
                    N_states = np.sum(~np.isnan(X_states[j,:]))
                    if N_states>2:
                        for it in range(2,N_states):
                            if X_states[j,it-2]==state1 and X_states[j,it-1]==state2 and X_states[j,it]==state3:
                                prob_third_sum[state1,state2,state3] += 1
    
    prob_third = np.zeros((2,2,2))
    for state1 in [0,1]:
        for state2 in [0,1]:
            prob_third[state1,state2,:] = prob_third_sum[state1,state2,:]/np.sum(prob_third_sum[state1,state2,:])
    
    return prob_first,prob_second,prob_third


def calc_allstays(X,delta_t):
    allstays = []
    
    N_part = X.shape[0]
    N_t = X.shape[1]
    
    for j in range(0,N_part):

        array12_j = [1 if x < 0 else 2 for x in X[j,:,0]]
        
        diffcellpos = np.diff(array12_j)
        N_transitions = len(np.where(diffcellpos != 0)[0])
        if N_transitions > 1:
            jumplocations = np.where(diffcellpos != 0)[0]
            startofmeas = jumplocations[0]+1
            endofmeas = jumplocations[-1]+1
            
            array12_j[:startofmeas] = [0] * startofmeas
            array12_j[endofmeas:] = [0] * (N_t-endofmeas)
            
            a = 0
            while a < N_t-1:
                b = a
                while array12_j[b]==array12_j[a] and b<N_t-1:
                    b += 1

                allstays.append(b-a)
                a = b

    allstays_resc = np.array(allstays)*delta_t    
    return allstays_resc


def calc_allstays_LR(X,delta_t):
    
    allstays_neg = []
    allstays_pos = []
    
    N_part = X.shape[0]
    N_t = X.shape[1]
    
    for j in range(0,N_part):

        array12_j = [1 if x < 0 else 2 for x in X[j,:,0]]
        
        diffcellpos = np.diff(array12_j)
        N_transitions = len(np.where(diffcellpos != 0)[0])
        if N_transitions > 1:
            jumplocations = np.where(diffcellpos != 0)[0]
            startofmeas = jumplocations[0]+1
            endofmeas = jumplocations[-1]+1
            
            array12_j[:startofmeas] = [0] * startofmeas
            array12_j[endofmeas:] = [0] * (N_t-endofmeas)
            
            a = 0
            while a < N_t-1:
                b = a
                while array12_j[b]==array12_j[a] and b<N_t-1:
                    b += 1

                allstays.append(b-a)
                a = b

    allstays_resc = np.array(allstays)*delta_t    
    return allstays_resc


################################################################
#Fxv inference
def calc_Fxv(data,X_output,x_lim,v_lim,N_bins_x,N_bins_v,Nobs_min=1):

    bins_x = np.linspace(-x_lim,x_lim,N_bins_x)
    bins_v = np.linspace(-v_lim,v_lim,N_bins_v)
    delta_x = (bins_x[1]-bins_x[0])/2
    delta_v = (bins_v[1]-bins_v[0])/2
    
    a_sum_xv = np.zeros((N_bins_x,N_bins_v))
    
    N_xv = np.zeros((N_bins_x,N_bins_v))
    N = 0
    for j in range(0,data.N_part):        
        for t in range(0,data.N_t[j]-2):
            for b_x in range(0,N_bins_x):
                if(data.X[j,t+1] > bins_x[b_x]-delta_x and data.X[j,t+1] < bins_x[b_x] + delta_x):
                    for b_v in range(0,N_bins_v):
                        if(data.V[j,t] > bins_v[b_v]-delta_v and data.V[j,t] < bins_v[b_v] + delta_v):
                            if ~np.isnan(data.A[j,t]):
                                a_sum_xv[b_x][b_v] += X_output[j,t]
                                
                                N_xv[b_x][b_v] += 1
                                N += 1

    Fxv = np.zeros((N_bins_x,N_bins_v))
    phist_xv = np.zeros((N_bins_x,N_bins_v))

    for b_x in range(0,N_bins_x):
        for b_v in range(0,N_bins_v):
            if(N_xv[b_x][b_v]>=Nobs_min):
                Fxv[b_x][b_v] = a_sum_xv[b_x][b_v]/N_xv[b_x][b_v]

            else:
                Fxv[b_x][b_v] = np.nan

    if N>0:
        phist_xv = N_xv/N
        
    return bins_x,bins_v,Fxv,phist_xv


def calc_Fv(data,X_output,width_x,v_lim,N_bins_v,Nobs_min=1):

    bins_v = np.linspace(-v_lim,v_lim,N_bins_v)
    delta_v = (bins_v[1]-bins_v[0])/2
    
    a_sum_xv = np.zeros((N_bins_v))
    
    N_xv = np.zeros((N_bins_v))
    N = 0
    for j in range(0,data.N_part):        
        for t in range(0,data.N_t[j]-2):
            if(data.X[j,t+1] > -width_x and data.X[j,t+1] < width_x):
                for b_v in range(0,N_bins_v):
                    if(data.V[j,t] > bins_v[b_v]-delta_v and data.V[j,t] < bins_v[b_v] + delta_v):
                        if ~np.isnan(data.A[j,t]) and data.A[j,t]<1e4:
                            a_sum_xv[b_v] += X_output[j,t]
                            
                            N_xv[b_v] += 1
                            N += 1

    Fxv = np.zeros((N_bins_v))
    phist_xv = np.zeros((N_bins_v))

    for b_v in range(0,N_bins_v):
        if(N_xv[b_v]>=Nobs_min):
            Fxv[b_v] = a_sum_xv[b_v]/N_xv[b_v]

        else:
            Fxv[b_v] = np.nan

    if N>0:
        phist_xv = N_xv/N
        
    return bins_v,Fxv,phist_xv




def calc_Fv_snip(X,v_lim,N_bins_v,delta_t,Nobs_min=1):
    
    V = np.diff(X)/delta_t
    A = np.diff(V)/delta_t
    N_part = X.shape[0]
    #N_t = X.shape[1]

    bins_v = np.linspace(-v_lim,v_lim,N_bins_v)
    delta_v = (bins_v[1]-bins_v[0])/2
    
    a_sum_xv = np.zeros((N_bins_v))
    
    N_xv = np.zeros((N_bins_v))
    N = 0
    for j in range(0,N_part):   
        N_t_here = len([x for x in X[j,:] if not np.isnan(x)])
        for t in range(0,N_t_here-2):
            for b_v in range(0,N_bins_v):
                if(V[j,t] > bins_v[b_v]-delta_v and V[j,t] < bins_v[b_v] + delta_v):
                    if ~np.isnan(A[j,t]) and A[j,t]<1e4:
                        a_sum_xv[b_v] += A[j,t]
                        
                        N_xv[b_v] += 1
                        N += 1

    Fxv = np.zeros((N_bins_v))
    phist_xv = np.zeros((N_bins_v))

    for b_v in range(0,N_bins_v):
        if(N_xv[b_v]>=Nobs_min):
            Fxv[b_v] = a_sum_xv[b_v]/N_xv[b_v]

        else:
            Fxv[b_v] = np.nan

    if N>0:
        phist_xv = N_xv/N
        
    return bins_v,Fxv,phist_xv







def calc_Fv_EC(data,X_output,width_x,v_lim,N_bins_v,CSI_crit,Nobs_min=1):

    bins_v = np.linspace(-v_lim,v_lim,N_bins_v)
    delta_v = (bins_v[1]-bins_v[0])/2
    
    a_sum_xv = np.zeros((N_bins_v,2))
    
    N_xv = np.zeros((N_bins_v,2))
    N = 0
    for j in range(0,data.N_part):        
        for t in range(0,data.N_t[j]-2):
            mode_comp = int(data.cellshapeindex[j,t+1] > CSI_crit)
            if(data.X[j,t+1] > -width_x and data.X[j,t+1] < width_x):
                for b_v in range(0,N_bins_v):
                    if(data.V[j,t] > bins_v[b_v]-delta_v and data.V[j,t] < bins_v[b_v] + delta_v):
                        if ~np.isnan(data.A[j,t]) and data.A[j,t]<1e4:
                            a_sum_xv[b_v,mode_comp] += X_output[j,t]
                            N_xv[b_v,mode_comp] += 1
                            N += 1

    Fxv = np.zeros((N_bins_v,2))
    phist_xv = np.zeros((N_bins_v,2))

    for b_v in range(0,N_bins_v):
        for mode_comp in [0,1]:
            if(N_xv[b_v,mode_comp]>=Nobs_min):
                Fxv[b_v,mode_comp] = a_sum_xv[b_v,mode_comp]/N_xv[b_v,mode_comp]
            else:
                Fxv[b_v,mode_comp] = np.nan

    if N>0:
        phist_xv = N_xv/N
        
    return bins_v,Fxv,phist_xv

def calc_sigmaxv(data,bins_x,bins_v,Fxv,delta_t,Nobs_min=1):

    delta_x = (bins_x[1]-bins_x[0])/2
    delta_v = (bins_v[1]-bins_v[0])/2
    N_bins_x = len(bins_x)
    N_bins_v = len(bins_v)
    
    a_sum_xv = np.zeros((N_bins_x,N_bins_v))
    a_sum = 0
    
    N_xv = np.zeros((N_bins_x,N_bins_v))
    N = 0
    for j in range(0,data.N_part):        
        for t in range(0,data.N_t[j]-2):
            for b_x in range(0,N_bins_x):
                if(data.X[j,t+1] > bins_x[b_x]-delta_x and data.X[j,t+1] < bins_x[b_x] + delta_x):
                    for b_v in range(0,N_bins_v):
                        if(data.V[j,t] > bins_v[b_v]-delta_v and data.V[j,t] < bins_v[b_v] + delta_v):
                            if ~np.isnan(Fxv[b_x][b_v]) and ~np.isnan(data.A[j,t]):
                                difference = data.A[j,t] - Fxv[b_x][b_v]
                                a_sum_xv[b_x][b_v] += difference**2
                                a_sum += difference**2
                                
                                N_xv[b_x][b_v] += 1
                                N += 1

    sigmaxv = np.zeros((N_bins_x,N_bins_v))
   
    for b_x in range(0,N_bins_x):
        for b_v in range(0,N_bins_v):
            if(N_xv[b_x][b_v]>=Nobs_min):
                sigmaxv[b_x][b_v] = np.sqrt(delta_t*a_sum_xv[b_x][b_v]/N_xv[b_x][b_v])
            else:
                sigmaxv[b_x][b_v] = np.nan
    if(N>0):
        sigma0 =  np.sqrt(delta_t*a_sum/N)
    else:
        sigma0 = np.nan
    
    return sigmaxv,sigma0


################################################################
def plot_movie_v2(X,P,state,L,a,subdirectory_plot_movie,j=0):
    from matplotlib.patches import Rectangle
    
    colors_CSI = [np.array([255,176,75])/255,
        np.array([169,94,255])/255,'grey']
    
    N_t_max = X.shape[1]

    N_t_max_movie = int(N_t_max)
    N_pics = int(N_t_max_movie)
    N_framerate = int(N_t_max_movie/N_pics)
    
    length = 2*a+L
    width = 6

    
    
    if a>0:
        fig_size = [np.round(1.2*length*0.03,2),2*a*0.03]   
    else:
        fig_size = [12,1]   
    params_plt = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params_plt)
    

    plt.close('all')
    #https://stackoverflow.com/questions/60620345/create-video-from-python-plots-to-look-like-a-moving-probability-distribution
    fig = plt.figure()
    ax = fig.gca()
    w=0 #vid frame step
    for t in range(0,N_t_max_movie,N_framerate):
        
        if a>0:
            ax.add_patch(Rectangle((-(a+L/2), -a/2), a, a, facecolor='lightgrey'))
            ax.add_patch(Rectangle((L/2, -a/2), a, a, facecolor='lightgrey'))
            ax.add_patch(Rectangle((-L/2, -width/2), L, width, facecolor='lightgrey'))
        else:
            ax.add_patch(Rectangle((-L/2, -width/2), L, width, facecolor='lightgrey'))
        
        
        col = 'dimgrey' #'green' #
        
        
        
        ax.scatter(X[j,t,0],0,400, color=colors_CSI[int(state[j,t])],edgecolors=col)
        ax.scatter(X[j,t,0],0,50, color=col)
        
        #fac = 75
        #ax.quiver(X[j,t,0],0,P[j,t,0]/fac,P[j,t,1]/fac,scale=7,color=col,width=0.005,headwidth=2)
        
        #Pnorm = np.sqrt(P[j,t,0]**2 + P[j,t,1]**2)
        #ax.quiver(X[j,t,0],0,P[j,t,0]/Pnorm,P[j,t,1]/Pnorm,scale=7,color=col,width=0.008,headwidth=2)
        
        Pmax = 1
        fac = 75
        
        Px = P[j,t,0]/fac
        Py = P[j,t,1]/fac
        
        Pnorm = np.sqrt(Px**2 + Py**2)
        
        if Pnorm>1:
            Px_plot = Px/Pnorm
            Py_plot = Py/Pnorm
        else:
            Px_plot = Px
            Py_plot = Py
        
        ax.quiver(X[j,t,0],0,Px_plot,Py_plot,scale=7,color=col,width=0.008,headwidth=2)
        
        

        
        x_lim_neg = -1.2*length/2
        x_lim_pos = 1.2*length/2
        
        plt.xlim([x_lim_neg,x_lim_pos])

        if a>0:
            plt.ylim([-a,a])
        else:
            plt.ylim([-2*width,2*width])
        
        """
        plt.plot(length/2*np.ones(2),[-y_line,y_line],'-k')
        plt.plot(-length/2*np.ones(2),[-y_line,y_line],'-k')
        plt.plot(L/2*np.ones(2),[-y_line,y_line],':k')
        plt.plot(-L/2*np.ones(2),[-y_line,y_line],':k')
        """
        
        plt.xticks([])
        plt.yticks([])
        
        #plt.axis('off')
        
        plt.savefig(subdirectory_plot_movie + '/' + f'plot_step_{w:04d}.png')
        plt.cla()
        
        w += 1
        
        
    os.chdir(subdirectory_plot_movie)
    print(subdirectory_plot_movie)
    if(os.path.isfile("movie.mp4")):
        os.remove("movie.mp4")
    import subprocess
    subprocess.call([
            'ffmpeg', '-framerate', '12', '-i', 'plot_step_%04d.png', '-r', '30', '-pix_fmt', 'yuv420p',
            'movie.mp4'
        ])
    
    #change back to top directory
    os.chdir('../../../..')
    
    
    return 0



def plot_movie_multi(X,P,state,L,a,N,subdirectory_plot_movie,j=0,x_size=1):
    from matplotlib.patches import Rectangle
    
    colors_CSI = [np.array([255,176,75])/255,
        np.array([169,94,255])/255,'grey']
    
    N_t_max = X.shape[1]

    N_t_max_movie = int(N_t_max)
    N_pics = int(N_t_max_movie)
    N_framerate = int(N_t_max_movie/N_pics)
    
    length = 2*a+L
    width = 6

    fig_size = [13,x_size]#np.round(a/32,2)]   
    params_plt = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params_plt)
    

    plt.close('all')
    #https://stackoverflow.com/questions/60620345/create-video-from-python-plots-to-look-like-a-moving-probability-distribution
    fig = plt.figure()
    ax = fig.gca()
    w=0 #vid frame step
    for t in range(0,N_t_max_movie,N_framerate):
        
        for i in range(-N,N+1,1): 
            if i > 0:
                ax.add_patch(Rectangle(((i-1)*a+L*(i-1/2), -a/2), a, a, facecolor='lightgrey'))
            elif i < 0:
                ax.add_patch(Rectangle((i*a+L*(i+1/2), -a/2), a, a, facecolor='lightgrey'))
            #ax.add_patch(Rectangle((L/2, -a/2), a, a, facecolor='lightgrey'))
            #ax.add_patch(Rectangle((-L/2, -width/2), L, width, facecolor='lightgrey'))

        col = 'dimgrey'

        ax.scatter(X[j,t,0],X[j,t,1],400, color=colors_CSI[int(state[j,t])],edgecolors=col)
        ax.scatter(X[j,t,0],X[j,t,1],50, color=col)

        Pmax = 1
        fac = 75
        
        Px = P[j,t,0]/fac
        Py = P[j,t,1]/fac
        
        Pnorm = np.sqrt(Px**2 + Py**2)
        
        if Pnorm>1:
            Px_plot = Px/Pnorm
            Py_plot = Py/Pnorm
        else:
            Px_plot = Px
            Py_plot = Py
        
        ax.quiver(X[j,t,0],X[j,t,1],Px_plot,Py_plot,scale=15,color=col,width=0.008,headwidth=2)
        
        x_lim_neg = -3*length
        x_lim_pos = 3*length
        
        plt.xlim([x_lim_neg,x_lim_pos])

        if a>0:
            plt.ylim([-a,a])
        else:
            plt.ylim([-2*width,2*width])

        plt.xticks([])
        plt.yticks([])
        
        #plt.axis('off')
        
        plt.tight_layout()
        
        plt.savefig(subdirectory_plot_movie + '/' + f'plot_step_{w:04d}.png')
        plt.cla()
        
        w += 1
        
        
    os.chdir(subdirectory_plot_movie)
    print(subdirectory_plot_movie)
    if(os.path.isfile("movie.mp4")):
        os.remove("movie.mp4")
    import subprocess
    subprocess.call([
            'ffmpeg', '-framerate', '12', '-i', 'plot_step_%04d.png', '-r', '30', '-pix_fmt', 'yuv420p',
            'movie.mp4'
        ])
    
    #change back to top directory
    os.chdir('../../../..')
    
    
    return 0

"""
################################################################
def plot_movie(X,P,L,a,subdirectory_plot_movie,j=0):
    from matplotlib.patches import Rectangle
    
    
    N_t_max = X.shape[1]

    N_t_max_movie = int(N_t_max)
    N_pics = int(N_t_max_movie)
    N_framerate = int(N_t_max_movie/N_pics)
    
    length = 2*a+L
    width = 6

    
    
    if a>0:
        fig_size = [np.round(1.2*length*0.03,2),2*a*0.03]   
    else:
        fig_size = [12,1]   
    params_plt = {
              'figure.figsize': fig_size,
              }
    plt.rcParams.update(params_plt)
    

    plt.close('all')
    #https://stackoverflow.com/questions/60620345/create-video-from-python-plots-to-look-like-a-moving-probability-distribution
    fig = plt.figure()
    ax = fig.gca()
    w=0 #vid frame step
    for t in range(0,N_t_max_movie,N_framerate):
        
        if a>0:
            ax.add_patch(Rectangle((-(a+L/2), -a/2), a, a, facecolor='lightgrey'))
            ax.add_patch(Rectangle((L/2, -a/2), a, a, facecolor='lightgrey'))
            ax.add_patch(Rectangle((-L/2, -width/2), L, width, facecolor='lightgrey'))
        else:
            ax.add_patch(Rectangle((-L/2, -width/2), L, width, facecolor='lightgrey'))
        
        Pnorm = np.sqrt(P[j,t,0]**2 + P[j,t,1]**2)
        ax.quiver(X[j,t,0],0,P[j,t,0]/Pnorm,P[j,t,1]/Pnorm,scale=7,color='fuchsia',width=0.01,headwidth=2)
        ax.scatter(X[j,t,0],0,200, color='green')

        
        x_lim_neg = -1.2*length/2
        x_lim_pos = 1.2*length/2
        
        plt.xlim([x_lim_neg,x_lim_pos])

        if a>0:
            plt.ylim([-a,a])
        else:
            plt.ylim([-2*width,2*width])
        
        
        plt.xticks([])
        plt.yticks([])
        
        #plt.axis('off')
        
        plt.savefig(subdirectory_plot_movie + '/' + f'plot_step_{w:04d}.png')
        plt.cla()
        
        w += 1
        
        
    os.chdir(subdirectory_plot_movie)
    print(subdirectory_plot_movie)
    if(os.path.isfile("movie.mp4")):
        os.remove("movie.mp4")
    import subprocess
    subprocess.call([
            'ffmpeg', '-framerate', '8', '-i', 'plot_step_%04d.png', '-r', '30', '-pix_fmt', 'yuv420p',
            'movie.mp4'
        ])
    
    #change back to top directory
    os.chdir('../../../..')
    
    
    return 0
"""

def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3] 
    subax = fig.add_axes([x,y,width,height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def readout_data_expt(subdirectory_data,length,mode,prefix=''):
    if len(prefix)>0:
        prefix = '_' + prefix
    f = open(subdirectory_data+'/data' + prefix + '_length' + str(length) + '_' + mode + ".pickle",'rb')
    data = pickle.load(f)
    f.close()
    return data





"""
def snippets_states_old(X,cellshapeindex,N_snippets_max,L,N_snip_per_traj_max=50,CSI_crit=0.4):
    
    N_part = X.shape[0]
    N_t_max = X.shape[1]
    
    X_snippets = np.zeros((N_snippets_max,N_t_max,2))
    X_snippets[:] = np.nan

    X_states = np.zeros((N_part,N_snip_per_traj_max))
    X_states[:] = np.nan
    
    cnt=0
    for j in range(0,N_part):
        
        N_t = len([x for x in X[j,:] if not np.isnan(x)])
        
        array12_j = np.zeros(N_t)
        for t in range(0,N_t):
            if np.abs(X[j,t]) < L/2:
                array12_j[t] = 1
            elif np.abs(X[j,t]) > L/2:
                array12_j[t] = 2
            else:
                array12_j[t] = np.nan
    
        diffcellpos = np.diff(array12_j)
        N_transitions = len(np.where(diffcellpos != 0)[0])
        if N_transitions > 1:
            jumplocations = np.where(diffcellpos != 0)[0]+1
            
            indices_err = np.where(np.array(jumplocations)>N_t-1)[0]
            if indices_err>0:
                del jumplocations[indices_err[0]]
            
            jumplocations = np.concatenate((np.array([0]), jumplocations, np.array([N_t-1])))
            N_jumps = len(jumplocations)
            
            cnt_states=0
            for it in range(0,N_jumps-1):
                snippet = X[j,jumplocations[it]:jumplocations[it+1]]
                snippet_CSI = cellshapeindex[j,jumplocations[it]:jumplocations[it+1]]
                
                if array12_j[jumplocations[it]]==1 and len(snippet)>5:
                    fac = -np.sign(snippet[0])
                    X_snippets[cnt,:len(snippet),0] = fac*snippet
                    X_snippets[cnt,:len(snippet),1] = snippet_CSI
                    
                    mean_CSI = np.nanmean(snippet_CSI)
                    X_states[j,cnt_states] = mean_CSI>CSI_crit
                    
                    cnt+=1
                    cnt_states+=1

    return X_snippets,X_states
"""