import warnings
from sklearn.decomposition import PCA, KernelPCA
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
import numpy as np  


from sklearn.model_selection import GridSearchCV

class KernelPCADenoiser(RegressorMixin, BaseEstimator):
    """
    Two-Photon movie denoiser class.
    """

    def __init__(
        self, 
        n_components=512,
        k=5, 
        alpha=0.01,
        kernel='rbf', 
        n_jobs=None,
        scaling='features', 
        reconstruction_method='mean',
        nonneg=False, 
        decimate=None,
    ):
        """
        A class for denoising 2-photon calcium imaging movie data using NMF (Non-negative matrix factorization).

        Parameters
        ----------
         n_components (int): the number of components to extract
         (int): the sparsity of the component's loadings
         alpha (float): regularization strength for sparsity
        kernel (str): kernel used for NMF
         n_jobs (int or None): number of parallel jobs to run, None means using all processors
        scaling (str): scaling method for the data matrix
        reconstruction_method (str): method used for reconstructing the denoised data
        nonneg (bool): whether to enforce non-negativity on the factors
        decimate (int or None): factor by which to reduce the size of the data along each dimension
 
        """
        self.n_components = n_components
        self.alpha = alpha
        self.kernel = kernel
        self.k = k
        self.n_jobs = n_jobs
        self.scaling = scaling
        self.reconstruction_method = reconstruction_method
        self.nonneg = nonneg
        self.decimate = decimate
        
    def _format_X(self, X):
        """
        Format the input data `X` for processing by the denoising algorithm.
    
        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape `(n_samples, n_features)`.
    
        Returns
        -------
            numpy.ndarray
            Formatted input data of shape `(n_samples, n_features * k)` if `k > 1`,
            or the original input data `X` if `k == 1`.
        """
        if self.k > 1:
            X = np.hstack([X[i:X.shape[0]-self.k+i+1] for i in range(self.k)])
        return X
    
    def _reformat_X(self, X):
        """
        Reformat the input data X depending on the reconstruction method and the value of k.
    
        If k == 1, return X as is.
        If reconstruction_method is 'mean', average the k frames around each timepoint.
        If reconstruction_method is 'middle', return a new array by stacking k/2 frames at the start and end of X, 
        then the middle k-2 frames of X.
    
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data to reformat.
        
        Returns:
        --------
        Xnew : array-like, shape (n_samples_new, n_features_new)
            The reformatted data X.
        """
        if self.k == 1:
            return X
        elif self.reconstruction_method == 'mean':
            count = np.zeros(X.shape[0]+self.k-1)
            Xnew = np.zeros((X.shape[0]+self.k-1, self.n_features_), dtype=X.dtype)
            for i in range(X.shape[0]):
                for ik in range(self.k):
                    Xnew[i+ik] += X[i, ik*self.n_features_:(ik+1)*self.n_features_]
                    count[i+ik] += 1
            Xnew = Xnew / count[:, None]
            return Xnew
        elif self.reconstruction_method == 'middle':
            # make sure the newX has the same shape of X before stacking timepoints
            # get "badly" estimated frames from top and bottom
            topX = np.zeros((self.k//2, self.n_features_))
            bottomX = topX.copy()
            for i in range(self.k // 2):
                topX[i] = X[
                    i, 
                    i*self.n_features_:(i+1)*self.n_features_
                ]
                bottomX[-(i+1)] = X[
                    -(i+1), 
                    self.k*self.n_features_-(i+1)*self.n_features_
                    :self.k*self.n_features_-i*self.n_features_
                ]
            # get middle estimated frames
            X = X[:, (self.k // 2 * self.n_features_):((self.k // 2 + 1) * self.n_features_)]
            # stack all arrays
            X = np.vstack([bottomX, X, topX])
            return X
        else:
            raise NameError(f"Reconstruction method: {self.reconstruction_method}") 
        
    def fit(self, X, y=None, sample_weight=None, X2=None):
        """
        Fits the model to the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
            
        y : array-like of shape (n_samples, n_features), default=None
            Unused parameter.
            
        sample_weight : array-like of shape (n_samples,), default=None
            Unused parameter.
        
        X2 : array-like of shape (n_samples, n_features), default=None
            Additional training input samples to concatenate with `X`.
        
        Returns:
        --------
        self : object
            Returns an instance of the class.
        """
        assert sample_weight is None, "sample weight must be None"
        if y is None:
            y = X
        assert X.shape == y.shape, "X and y must be same shape"
        assert (self.k % 2) == 1, "k must be odd"
        if self.nonneg:
            assert np.all(X >= 0), "X must be non-negative"
        
        pca = KernelPCA(
            n_components=self.n_components, 
            kernel=self.kernel, 
            n_jobs=self.n_jobs, 
            alpha=self.alpha,
            fit_inverse_transform=True, 
            copy_X=False  # save memory
        )
        
        if self.scaling is None:
            pass
        elif self.scaling == 'features':
            self.scaler_ = StandardScaler(copy=False)
            X = self.scaler_.fit_transform(X)
        elif self.scaling == 'overall':
            self.mean_ = np.mean(X)
            self.std_ = np.std(X)
            X = (X - self.mean_)/self.std_
        else:
            raise NameError(f"Scaling parameter unknown {self.scaling}")
        
        if X2 is not None:
            print("Adding X2")
            self.x2_scaler_ = StandardScaler()
            X2 = self.x2_scaler_.fit_transform(X2)
            self.x2_features_ = X2.shape[1]
            X = np.hstack([X2, X])
        else:
            self.x2_features_ = 0
            
        self.n_features_ = X.shape[-1]
        
        X = self._format_X(X)
        
        if self.decimate is not None:
            X = X[::self.decimate]
        
        pca.fit(X)
        self.pca_ = pca
        
        return self
    
    def predict(self, X, X2=None):
        """
        Predict the reconstruction of the input data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to reconstruct.

        X2 : array-like of shape (n_samples, n_features_x2), default=None
            Additional input data to concatenate with X.

        Returns
        -------
        X_pred : ndarray of shape (n_samples, n_features)
        The reconstructed data.
        """
        if self.nonneg:
            assert np.all(X >= 0), "X must be non-negative"
        # scale inputs
        if self.scaling is None:
            pass
        elif self.scaling == 'features':
            X = self.scaler_.transform(X)
        elif self.scaling == 'overall':
            X = (X - self.mean_)/self.std_
        else:
            raise NameError(f"Scaling parameter unknown {self.scaling}")
        
        if self.x2_features_:
            assert X2 is not None, "X2 was used"
            X2 = self.x2_scaler_.fit_transform(X2)
            X = np.hstack([X2, X])
        # format X given self.k
        X = self._format_X(X)
        # prediction as transformation
        X = self.pca_.inverse_transform(self.pca_.transform(X))
        # format back to original
        X = self._reformat_X(X)
        
        if self.x2_features_:
            # cut out X2 features
            X = X[:, self.x2_features_:]

        # inversely transform X scaling
        if self.scaling is None:
            pass
        elif self.scaling == 'features':
            X = self.scaler_.inverse_transform(X)
        elif self.scaling == 'overall':
            X = (X * self.std_) + self.mean_
        else:
            raise NameError(f"Scaling parameter unknown {self.scaling}")
        # ensure non-negativity
        if self.nonneg:
            X[X < 0] = 0
        return X
    
    
class KernelRidgeDenoiser(RegressorMixin, BaseEstimator):
    """
    Two-Photon movie denoiser class using kernel ridge regression
    """

    def __init__(
        self, 
        n_components=None,
        reconstruction_method=None,
        k=5, 
        alpha=0.01,
        kernel='rbf', 
        scaling='features', 
        nonneg=False, 
        decimate=None,
        pad=True, 
        model='kernel',
        verbose=1
    ):
        self.n_components = n_components
        self.reconstruction_method = reconstruction_method
        self.alpha = alpha
        self.kernel = kernel
        self.k = k
        self.scaling = scaling
        self.nonneg = nonneg
        self.decimate = decimate
        self.pad = pad
        self.model = model
        self.verbose = verbose
        
    def _format_X(self, X):
        if self.verbose:
            print("assembling X")
        if self.pad:
            y = X
            X = np.vstack([
                np.zeros((self.k//2, X.shape[-1])), 
                X, 
                np.zeros((self.k//2, X.shape[-1])), 
            ])
        else:
            y = X[self.k//2:-self.k//2]
        
        X = np.hstack([
            X[i:X.shape[0]-self.k+i+1] for i in range(self.k)
            if i != (self.k//2)  # skip middle - to predict
        ])
        return X, y  
        
    def fit(self, X, y=None, sample_weight=None, X2=None):
        assert X2 is None, "X2 not implemented for kernel ridge"
        if self.n_components is not None:
            warnings.warn("n_components will be ignored for kernel ridge.")
        assert sample_weight is None, "sample weight must be None"
        if y is None:
            y = X
        assert X.shape == y.shape, "X and y must be same shape"
        assert (self.k % 2) == 1, "k must be odd"
        assert (self.k > 3), "k must be larget than 3"
        if self.nonneg:
            assert np.all(X >= 0), "X must be non-negative"
        
        if self.verbose:
            print('scaling X')
        if self.scaling is None:
            pass
        elif self.scaling == 'features':
            self.scaler_ = StandardScaler(copy=False)
            X = self.scaler_.fit_transform(X)
        elif self.scaling == 'overall':
            self.mean_ = np.mean(X)
            self.std_ = np.std(X)
            X = (X - self.mean_)/self.std_
        else:
            raise NameError(f"Scaling parameter unknown {self.scaling}")
        
        if self.model == 'kernel':
            if self.verbose:
                print("kernel model for prediction")
            ridge = KernelRidge(
                alpha=self.alpha, 
                gamma=1/(X.shape[-1]*(self.k-1)), 
                kernel=self.kernel
            )
        elif self.model == 'linear':
            if not self.alpha:
                if self.verbose:
                    print("linear model for prediction")
                ridge = LinearRegression(copy_X=False, n_jobs=-1)
            else:
                if self.verbose:
                    print("ridge model for prediction")
                ridge = Ridge(alpha=self.alpha, copy_X=False)
        elif self.model == 'svr':
            if self.verbose:
                print("svr model for prediction")
            assert self.scaling != 'features', "No feature scaling for SVR!!!"
            assert X.shape[0]/(1 if self.decimate is None else self.decimate) < 1000, "Too many samples for SVR"          
            ridge = SVR(
                kernel=self.kernel, 
                # tenth percentile is background for sure
                epsilon=np.percentile(np.std(X, axis=0), q=10), 
                C=1/self.alpha, 
                cache_size=200
            )
            ridge = MultiOutputRegressor(ridge, n_jobs=-1)
        else:
            raise NameError(f"AHHH no model {self.model}")
            
        self.n_features_ = X.shape[-1]
        X, y = self._format_X(X)
        
        if self.decimate is not None:
            if self.verbose:
                print(f"decimating by {self.decimate}")
            X = X[::self.decimate]
            y = y[::self.decimate]
        
        ridge.fit(X, y)
        self.ridge_ = ridge
        
        return self
    
    def predict(self, X, X2=None):
        assert X2 is None, "X2 not implemented for kernel ridge"
        if self.nonneg:
            assert np.all(X >= 0), "X must be non-negative"
        # scale inputs
        if self.scaling is None:
            pass
        elif self.scaling == 'features':
            X = self.scaler_.transform(X)
        elif self.scaling == 'overall':
            X = (X - self.mean_)/self.std_
        else:
            raise NameError(f"Scaling parameter unknown {self.scaling}")
        # format X given self.k
        X, _ = self._format_X(X)
        # prediction as transformation
        if self.verbose:
            print("predicting")
        X = self.ridge_.predict(X)
        if self.verbose:
            print('rescaling')

        # inversely transform X scaling
        if self.scaling is None:
            pass
        elif self.scaling == 'features':
            X = self.scaler_.inverse_transform(X)
        elif self.scaling == 'overall':
            X = (X * self.std_) + self.mean_
        else:
            raise NameError(f"Scaling parameter unknown {self.scaling}")
        # ensure non-negativity
        if self.nonneg:
            X[X < 0] = 0
        return X
            
            
            
            
        
def denoiseSlice(slice,n_comp=64,a=0.01,k=15):
    definition = """
    denoise a slice from pv using kernelpca
    ---
    slice: np.array of imaging stack in T x Y x X
    """
    movie = slice
    kws = {
    # hyperparameters to tune during cross-validation
    # GridSearchCV
    'n_components': [n_comp], 
    'alpha': [a], 
    'k': [k],
    # how to cross-validate
    'cv': 2,
    # number of jobs
    'n_jobs': 5,
    # validation scoring
    'scoring': ['neg_mean_squared_log_error', 'neg_mean_absolute_error', 'neg_mean_squared_error'], 
    # preprocessing for KernelPCA
    # 'scaling': ['overall', 'features'], # overall or features
    'scaling': ['overall'],
    # How to reconstruct the whole image
    # 'reconstruction_method': ['mean', 'middle'],  # mean or middle
    'reconstruction_method': ['mean'],
    'nonneg': True, 
    'decimate': None, 
    'ridge': False, 
    'model_kwargs': {}
    }

    if kws['nonneg']:
        movie = movie.copy()
        movie[movie < 0] = 0
            
    print("setting up denoiser object")
    if kws['ridge']:
        est = KernelRidgeDenoiser(
            k=kws['k'][0], 
            alpha=kws['alpha'][0], 
            scaling=kws['scaling'][0], 
            nonneg=kws['nonneg'], 
            decimate=kws['decimate'], 
            **kws['model_kwargs']
        ) 
    else:
        est = KernelPCADenoiser(
            n_components=kws['n_components'][0], 
            k=kws['k'][0], 
            alpha=kws['alpha'][0], 
            scaling=kws['scaling'][0], 
            reconstruction_method=kws['reconstruction_method'][0], 
            nonneg=kws['nonneg'], 
            decimate=kws['decimate'], 
            **kws['model_kwargs']
        )

    do_gridsearch = np.any(np.array([
        len(v) for k, v in kws.items()
        if k in ['reconstruction_method', 'scaling', 'alpha', 'k', 'n_components']
    ]) > 1)

    if do_gridsearch:
        print("doing gridsearch")
        param_grid = {
            k: v for k, v in kws.items()
            if k in ['reconstruction_method', 'scaling', 'alpha', 'k', 'n_components']
        }
        
        gridcv = GridSearchCV(
            est, param_grid, scoring=kws['scoring'], n_jobs=kws['n_jobs'], 
            cv=kws['cv'], verbose=2, 
            refit=kws['scoring'][0]
        )
        
        X = movie.reshape(movie.shape[0], -1)
        print("finding and fitting best denoising model")
        gridcv.fit(X, X)
        print("denoising movie")
        X = gridcv.predict(X)
        
        movie = X.reshape(movie.shape)
        metadata = {
            'cv_results': gridcv.cv_results_, 
            'best_score': gridcv.best_score_, 
            'best_params': gridcv.best_params_, 
        }
    else:
        print("Not doing gridsearch as only one option")
        X = movie.reshape(movie.shape[0], -1)
        est.n_jobs = kws['n_jobs']
        print("fitting denoising model")
        est.fit(X, X)
        print("denoising movie")
        X = est.predict(X)
        movie = X.reshape(movie.shape)
        metadata = {}
    return movie


def denoiseStack(zstack, n_comp=64,a=0.01,k=15):
    definition = """
    denoise a z-stack from pv using kernel pc
    ---
    zstack: np.array of imaging stack in Z x T x Y x X
    k:  For reconstruction. If k == 1, return X as is.
        If reconstruction_method is 'mean', average the k frames around each timepoint.
        If reconstruction_method is 'middle', return a new array by stacking k/2 frames at the start and end of X, 
        then the middle k-2 frames of X.
    """
    dstack =[]
    for i in range(0,np.shape(zstack)[0]):
        dSlice = denoiseSlice(zstack[i],n_comp=n_comp,a=a,k=k)
        dstack.append(dSlice)
    return np.array(dstack)
