### Basic imports
import numpy as np
import matplotlib.pyplot as plt
from getdist.gaussian_mixtures import GaussianND
from getdist import plots, MCSamples
from numdifftools import Hessian, Jacobian
from pandas import DataFrame



### Small helper functions
def Limit(limits, variable, mu, sig) :
    if not (variable in limits) :
        return (mu - 3 * sig, mu + 3 * sig)
    low_lim = max(mu - 3 * sig, limits[variable][0])
    up_lim = min(mu + 3 * sig, limits[variable][1])
    return (low_lim, up_lim)

def inverse(mat) :
    if len(mat.shape) == 2 :
        return np.linalg.inv(mat)
    return 1.0 / mat



### The actual class
class FisherMatrix :
    """
    Universal class for Fisher matrices. Allows adding matrices, including priors and marginalization. Plots are based on getdist.
    Includes the option to estimate the Fisher matrix from a likelihood using the Hessian.
   
    This class works well for a moderate number of parameters (< 20), thereafter it might become slow.
    """
   
    # Initilize Matrix
    def __init__(self, names, mean, matrix=None, from_cov=False,
                 limits={}, label="Fisher", labels=None, log_likelihood=None) :
        """
        names :          list of the names of the variables constrained by the fisher matrix
       
        labels :         labels of variables, such that it can be passed to getdist. If None, uses names.
       
        mean :           np.array of shape names.size, fiducial values of the variables
       
        matrix :         if given, np.array of shape (names.size, names.size), such that
                         fisher_matrix[i, j] = Fisher Information(names[i], names[j]) if from_cov == False (default)
                         otherwise matrix is the covariance matrix.
       
        label:           name of the Fisher matrix, default is Fisher
       
        limits:          dictonary of hard limits for each parameter, limits[names[xi]] = (xi_min, xi_max)
       
        log_likelihood : log likelihood from which the Fisher is esimated via the Hessian: Fij = - < d^2( log likelihood )/(dxi dxj) >
                         Hessian is computed using numdifftools (finite differences). Expectation value is approximated by
                         evaluation at mean (exact for Gaussian posteriors).
                         
        Note that self.samples is a getdist GaussianND ready for plotting. If limits exist, self.sample()
        should be called instead to generate samples with limits imposed.
        """
        self.names = np.array(names)
        if self.names.size > np.unique(self.names).size :
            raise ValueError("Initilization Error: names must be unique.")
       
        if labels is None :
            labels = self.names
        self.labels = labels
       
        self.mean = np.array(mean)
        self.label = label
        self.limits = limits
       
        if ((matrix is None) and (log_likelihood is None)) or ((matrix is not None) and (log_likelihood is not None)):
            raise ValueError("Initilization Error: must pass either Fisher matrix or log likelihood.")
        elif log_likelihood is None :
            if from_cov :
                self.fisher_matrix = inverse(np.array(matrix)); self.cov = np.array(matrix)
            else :
                self.fisher_matrix = np.array(matrix); self.cov = inverse(self.fisher_matrix)
        else :
            self.fisher_matrix = - Hessian(log_likelihood)(self.mean)
            self.cov = inverse(self.fisher_matrix)
       
        self.samples = GaussianND(mean=self.mean, cov=self.fisher_matrix, is_inv_cov=True,
                                  names=self.names, labels=self.labels, label=self.label)
       
    # Plots (one variable, 2 variables or total)
    def plot(self, variables=None, out_file=None, fontsize=15, fig_width=6.8, ratio=0.75, shade1d=False, other=None):
        """
        Produces a corner plot of variables. If variables is 1d, it produces a plot of the density. If variables is 2d,
        produce a plot of the corresponding contour. Otherwise, show the full corner plot.
       
        If out_file is not none, the plot is saved there.
        """
        if variables is None :
            variables = self.names
       
        # Font and init figure
        plt.rcParams.update({"font.size": fontsize, "axes.labelsize": fontsize,
                             "xtick.labelsize": fontsize, "ytick.labelsize": fontsize,
                             "legend.fontsize": fontsize, "axes.titlesize": fontsize,
                             "font.family": "serif", "text.usetex": True})
        plt.figure(dpi=250, figsize=(fig_width, ratio * fig_width))
       
        # 1d case
        if (len(variables) == 1) and (len(self.names) != 1) :
            g = plots.get_single_plotter(width_inch=fig_width, rc_sizes=True)
            g.plot_1d(self.samples, variables[0])
            index = np.where(self.names == variables[0])[0][0]
            plt.xlim([*Limit(self.limits, variables[0], self.mean[index], np.sqrt(self.cov[index, index]))])
           
        # 2d case    
        elif (len(variables) == 2) and (len(self.names) != 2) :
            g = plots.get_single_plotter(ratio=1, width_inch=fig_width, rc_sizes=True)
            g.plot_2d(self.samples, variables, filled=True)
            plt.grid(True, alpha=0.25)
            index_x = np.where(self.names == variables[0])[0][0]
            index_y = np.where(self.names == variables[1])[0][0]
            plt.xlim([*Limit(self.limits, variables[0], self.mean[index_x], np.sqrt(self.cov[index_x, index_x]))])
            plt.ylim([*Limit(self.limits, variables[1], self.mean[index_y], np.sqrt(self.cov[index_y, index_y]))])
           
        # full case
        else :
            param_limits = {self.names[i] : Limit(self.limits, variables[i], self.mean[i], np.sqrt(self.cov[i, i]))
                            for i in range(len(self.names)) }
            g = plots.get_subplot_plotter(width_inch=fig_width, rc_sizes=True)

            if other is None:
                g.triangle_plot(self.samples, filled=True, param_limits=param_limits)
            
            else:
                shade1d = False

                g.triangle_plot([self.samples, other.samples], filled=[False,False], param_limits = param_limits)


            if shade1d:
                sigmas = np.sqrt(np.diag(self.cov))
                for i, name in enumerate(self.names):
                    ax = g.subplots[i, i]  # diagonal 1D panel

                    line = ax.get_lines()[0]
                    x_data, y_data = line.get_data() 

                    alphas = [2./3, 1./3]

                    for j in range(2):

                        xmin = self.mean[i] - (j+1)*sigmas[i]
                        xmax = self.mean[i] + (j+1)*sigmas[i]

                        x_fill = np.linspace(xmin, xmax, 100)

                        y_fill = np.interp(x_fill, x_data, y_data)

                        color = g.settings.solid_colors[0]

                        ax.fill_between(x_fill, y_fill, color=color, alpha = alphas[j])

                        if j == 0:
                            ax.vlines(xmin, ymin = 0., ymax = y_fill[0], color='black', linestyle='--')
                            ax.vlines(xmax, ymin = 0., ymax = y_fill[-1], color='black', linestyle='--')
                        
       
        # Save if wanted and show
        if out_file != None :
            plt.savefig(out_file)
        plt.show()
        plt.close()
       
    # Produce samples from the corresponding Gaussian
    def sample(self, N=100000) :
        """
        Generate an array of N independent samples from the gaussian likelihood defined by the Fisher, with limits imposed.
       
        Return shape is (len(names), N_with_limts), with samples[i, :] sampling the variable names[i].
        Note that when limits are imposed N_with_limts < N in general.
        """
        samples = np.random.multivariate_normal(mean=self.mean, cov=self.cov, size=N)

        # Impose limits
        for i in range(len(self.names)) :
            if self.names[i] in self.limits :
                sample_i = samples[:, i]
                mask_i = (self.limits[self.names[i]][0] < sample_i) & (sample_i < self.limits[self.names[i]][1])
                samples = samples[mask_i, :]
       
        return samples.T

    # Marginalize over variables
    def marginalize(self, variables) :
        """
        variables is a list of variables which are marginalized over analytically (marginalization ignores limits).
        """
        # Containers for New Variables
        variables = np.array(variables)
        new_size = len(self.names) - len(variables)
        new_names = np.setdiff1d(self.names, variables, assume_unique=True)
        new_labels = np.zeros_like(new_names)
        new_cov = np.zeros((new_size, new_size))
        new_mean = np.ones(new_size)
       
        # Fill cov and compute Fisher
        for i,j in np.ndindex(new_cov.shape) :
            index_i = np.where(self.names == new_names[i])[0][0]
            index_j = np.where(self.names == new_names[j])[0][0]
           
            new_labels[i] = self.labels[index_i]
            new_mean[i] = self.mean[index_i]
            new_cov[i, j] = self.cov[index_i, index_j]
           
        # New Limits
        new_limits = {}
        for i in range(new_size) :
            if new_names[i] in self.limits :
                index_i = np.where(self.names == new_names[i])[0][0]
                new_limits[new_names[i]] = self.limits[self.names[index_i]]
           
        # Update
        self.names = new_names; self.labels = new_labels; self.mean = new_mean
        self.fisher_matrix = inverse(new_cov); self.cov = new_cov
        self.samples = GaussianND(mean=self.mean, cov=self.fisher_matrix, is_inv_cov=True,
                                 names=self.names, labels=self.labels, label=self.label)
        self.limits = new_limits
       
    # Add a Gaussian prior
    def prior(self, variables, cov_prior) :
        """
        variables is a list of variables for which a gaussian prior with covariance matrix cov_prior (np.array of shape
        (variables.shape, variables.shape) is added. This is equivalent to adding the corresponding Fisher matrix, but
        this call may be more convenient.
        """
        # Get prior in right shape
        cov_prior = np.array(cov_prior)
        prior_fisher = np.zeros_like(self.fisher_matrix); fisher_prior = inverse(cov_prior)
        if len(fisher_prior.shape) == 2 :
            for i,j in np.ndindex(fisher_prior.shape) :
                index_i = np.where(self.names == variables[i])[0][0]
                index_j = np.where(self.names == variables[j])[0][0]
                prior_fisher[index_i, index_j] = fisher_prior[i, j]
        else :
            index_i = np.where(self.names == variables[0])[0][0]
            prior_fisher[index_i, index_i] = fisher_prior[0]
       
        # Update relevant information
        self.fisher_matrix = self.fisher_matrix + prior_fisher; self.cov = inverse(self.fisher_matrix)
        self.samples = GaussianND(mean=self.mean, cov=self.fisher_matrix, is_inv_cov=True,
                                 names=self.names, labels=self.labels, label=self.label)
   
    # update functions
    def update_label(self, label) :
        self.label = label
        self.samples = GaussianND(mean=self.mean, cov=self.fisher_matrix, is_inv_cov=True,
                                 names=self.names, labels=self.labels, label=self.label)
    def update_names(self, names) :
        if self.labels == self.names :
            self.labels = names
        self.names = names
        self.samples = GaussianND(mean=self.mean, cov=self.fisher_matrix, is_inv_cov=True,
                                 names=self.names, labels=self.labels, label=self.label)
    def update_labels(self, labels) :
        self.labels = labels
        self.samples = GaussianND(mean=self.mean, cov=self.fisher_matrix, is_inv_cov=True,
                                 names=self.names, labels=self.labels, label=self.label)
    def update_mean(self, mean) :
        self.mean = mean
        self.samples = GaussianND(mean=self.mean, cov=self.fisher_matrix, is_inv_cov=True,
                                 names=self.names, labels=self.labels, label=self.label)
    def update_fisher(self, mat, is_cov=False) :
        """
        Change fisher matrix to mat. If is_cov = True, mat is the covariance matrix instead.
        """
        if is_cov :
            self.fisher_matrix = inverse(mat); self.cov = mat
            self.samples = GaussianND(mean=self.mean, cov=self.fisher_matrix, is_inv_cov=True,
                                      names=self.names, labels=self.labels, label=self.label)
        else :
            self.fisher_matrix = mat; self.cov = inverse(mat)
            self.samples = GaussianND(mean=self.mean, cov=self.fisher_matrix, is_inv_cov=True,
                                      names=self.names, labels=self.labels, label=self.label)
    def update_limits(self, limits, variable=None) :
        """
        if variable is None, limits is a dic. Otherwise limits = (variable_min, variable_max).
        """
        if variable is None :
            self.limits = limits
        else :
            self.limits[variable] = limits
       
    # printing options
    def pprint(self, print_cov=False) :
        if print_cov :
            data = DataFrame(self.cov, columns=self.names, index=self.names)
            print( data )
        else :
            data = DataFrame(self.fisher_matrix, columns=self.names, index=self.names)
            print( data )
    def print_params(self) :
        data = DataFrame(np.vstack((self.mean, np.sqrt(np.diag(self.cov)))).T, columns=["mean", "error"], index=self.names)
        print( data )
   
    # save Fisher
    def save(self, directory=None) :
        """
        saves the Fisher matrix in directory (default is "label.txt") as a .txt file with the names as header.
        """
        if directory is None :
            directory = self.label + ".txt"
        header = ''
        for i in range(len(self.names)) :
            header += self.names[i] + ', '
        np.savetxt(directory, self.fisher_matrix, header = header)
       
    # Small Helper Functions
    def determinant(self, cov_det=False) :
        if cov_det :
            return np.linalg.det(self.cov)
        else :
            return np.linalg.det(self.fisher_matrix)
    def correlation(self):
        """
        Returns the correlation matrix corresponding to the covariance matrix.
        """
        diag = np.sqrt(np.diag(self.cov))
        corr = self.cov / np.outer(diag, diag)
        return corr
   
    # Transform between variables
    def transform(self, transformation,
                new_names=None, new_labels=None, new_label=None) :
        """
        Here transformation is a function of the form
       
            transformation( current_variables ) -> new_variables
           
        where both current_variables and new_variables are arrays of the same size. This function is used to estimate the
        covariance matrix of the transformed variables, using a numerical approximation of the Jacobian.
       
        We note that limits are ignored for this and should be added a posteriori.  
        """
        # Preform Transformation
        new_mean = transformation( self.mean )
        J = Jacobian(transformation)(self.mean)
        self.fisher_matrix = J.T @ self.fisher_matrix @ J
        self.limits = {}
       
        # Update Fisher
        self.cov = inverse( self.fisher_matrix ); self.mean = new_mean
        if (not (new_names is None)) and (not (new_labels is None)):
            self.names = new_names
            self.labels = new_labels
        if not (new_names is None) :
            self.names = new_names
            self.labels = new_names
        if not (new_label is None) :
            self.label = new_label
   
    # Add two Fisher Matrices
    def __add__(self, other):
        """
        Adds to Fisher matrices, creating a new Fisher matrix, which holds information about the union of the two variable sets.
       
        Consistency between matrices (i.e. same mean) is not enforced, but still assumed.
        """
        # Containers
        names = np.union1d(self.names, other.names)
        labels = np.ones_like(names); mean = np.zeros(names.size)
        self_fisher = np.zeros((names.size, names.size)); other_fisher = np.zeros((names.size, names.size))
       
        # Fill Containers
        for i in range(names.size) :
            if (names[i] in self.names) :
                self_index_i = np.where(self.names == names[i])[0][0]
                labels[i] = self.labels[self_index_i]; mean[i] = self.mean[self_index_i]
            else :
                other_index_i = np.where(other.names == names[i])[0][0]
                labels[i] = other.labels[other_index_i]; mean[i] = other.mean[other_index_i]
               
        for i in range(names.size) :
            if (names[i] in self.names) :
                self_index_i = np.where(self.names == names[i])[0][0]
                for j in range(names.size) :
                    if (names[j] in self.names) :
                        self_index_j = np.where(self.names == names[j])[0][0]
                        self_fisher[i, j] = self.fisher_matrix[self_index_i, self_index_j]

        for i in range(names.size) :
            if (names[i] in other.names) :
                other_index_i = np.where(other.names == names[i])[0][0]
                for j in range(names.size) :
                    if (names[j] in other.names) :
                        other_index_j = np.where(other.names == names[j])[0][0]
                        other_fisher[i, j] = other.fisher_matrix[other_index_i, other_index_j]

        # Fill limits
        limits = {}
        for i in range(names.size) :
           
            if (names[i] in self.limits) and (names[i] in other.limits) :
                self_index_i = np.where(self.names == names[i])[0][0]
                other_index_i = np.where(other.names == names[i])[0][0]
                limits[names[i]] = (max(self.limits[self.names[self_index_i]][0], other.limits[other.names[other_index_i]][0]),
                                    min(self.limits[self.names[self_index_i]][1], other.limits[other.names[other_index_i]][1]))
               
            elif names[i] in self.limits :
                self_index_i = np.where(self.names == names[i])[0][0]
                limits[names[i]] = (self.limits[self.names[self_index_i]][0], self.limits[self.names[self_index_i]][1])
               
            elif names[i] in other.limits :
                other_index_i = np.where(other.names == names[i])[0][0]
                limits[names[i]] = (other.limits[other.names[other_index_i]][0], other.limits[other.names[other_index_i]][1])
       
        return FisherMatrix(names, mean, self_fisher + other_fisher, limits=limits,
                            label=self.label + " + " + other.label, labels=labels)
   
   
   
### Some functions that use the Fisher class
# plotting multiple matrices
def plot_multiple(Fishers, ignore_bounds=True, filled=True, colors=None, out_file=None,
                  fontsize=15, fig_width=6.8, ratio=0.75, sampling_size=1000000) :
    """
    Plot multiple Fisher matrices and save image in out_file (if given). Default is to ignore bounds, otherwise
    Fisher.sample() is called to create getdist samples with bounds of size sampling_size.  
    """
   
    # Font and init figure
    plt.rcParams.update({"font.size": fontsize, "axes.labelsize": fontsize,
                         "xtick.labelsize": fontsize, "ytick.labelsize": fontsize,
                         "legend.fontsize": fontsize, "axes.titlesize": fontsize,
                         "font.family": "serif", "text.usetex": True})
    plt.figure(dpi=250, figsize=(fig_width, ratio * fig_width))
   
    # plot without bounds
    if ignore_bounds :
        g = plots.get_subplot_plotter(width_inch=fig_width, rc_sizes=True)
        g.triangle_plot([ F.samples for F in Fishers ], filled=filled, contour_colors=colors)
   
    # plot with bounds
    else :
        samps = [ MCSamples(samples = F.sample(int(sampling_size)).T,
                            names = F.names, labels = F.labels, label = F.label) for F in Fishers ]
        g = plots.get_subplot_plotter(width_inch=fig_width, rc_sizes=True)
        g.triangle_plot(samps, filled=filled, contour_colors=colors)
   
    # Save if wanted and show
    if out_file != None :
        plt.savefig(out_file)
    plt.show()
    plt.close()
   
# plotting two variables for multiple matrices
def plot_multiple2d(Fishers, variables, ignore_bounds=True, filled=True, colors=None, out_file=None,
                    fontsize=15, fig_width=6.8, ratio=0.75, sampling_size=1000000) :
    """
    Plot variables for multiple Fisher matrices and save image in out_file (if given). Default is to ignore bounds, otherwise
    Fisher.sample() is called to create getdist samples with bounds of size sampling_size.
    """
   
    # Font and init figure
    plt.rcParams.update({"font.size": fontsize, "axes.labelsize": fontsize,
                         "xtick.labelsize": fontsize, "ytick.labelsize": fontsize,
                         "legend.fontsize": fontsize, "axes.titlesize": fontsize,
                         "font.family": "serif", "text.usetex": True})
    plt.figure(dpi=250, figsize=(fig_width, ratio * fig_width))
   
    # plot without bounds
    if ignore_bounds :
        g =  plots.get_single_plotter(ratio=1, width_inch=fig_width, rc_sizes=True)
        g.plot_2d([ F.samples for F in Fishers ], variables, filled=filled, contour_colors=colors)
        g.add_legend([ F.label for F in Fishers ], colored_text=False)
   
    # plot with bounds
    else :
        samps = [ MCSamples(samples = F.sample(int(sampling_size)).T,
                            names = F.names, labels = F.labels, label = F.label) for F in Fishers ]
        g =  plots.get_single_plotter(ratio=1, width_inch=fig_width, rc_sizes=True)
        g.plot_2d(samps, variables, filled=filled, contour_colors=colors)
        g.add_legend([ F.label for F in Fishers ], colored_text=False)
       
    # Save if wanted and show
    if out_file != None :
        plt.savefig(out_file)
    plt.show()
    plt.close()