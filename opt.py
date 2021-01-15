# Written by Andrew Ning.  Feb 2016.
# FLOW Lab, Brigham Young University.
# Adapted by B. Garcia Dec 2018
import numpy as np

try:
    import matlab.engine
    import matlab
except ImportError:
    import warnings

    warnings.warn("""
    Matlab engine not installed.
    Instructions here: http://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
    If still having problems, try setting DYLD_FALLBACK_LIBRARY_PATH to contain your python lib location.
    See: http://www.mathworks.com/matlabcentral/answers/233539-error-importing-matlab-engine-into-python
    """)


def start():
    global eng
    # already_exist = matlab.engine.find_matlab()
    # if len(already_exist):
    #     eng_name = already_exist[0]
    #     eng = matlab.engine.connect_matlab(eng_name)
    #     print(f"using {eng_name}")
    # else:
    eng = matlab.engine.start_matlab()


def stop():
    global eng
    eng.quit()
    del eng


def fmincon(function, x0, lb, ub, nonlcon=[], A=[], b=[], Aeq=[],
            beq=[], options={}, providegradients=False, optional_args=[],
            engine=None):
    global eng
    if engine is not None:
        if 'eng' in globals():
            del eng
        eng = engine

    # convert to numpy array then list then to matlab type
    # these first conversions are necessary to allow both numpy and list style inputs
    x0 = matlab.double(np.array(x0).tolist())
    ub = matlab.double(np.array(ub).tolist())
    lb = matlab.double(np.array(lb).tolist())
    A = matlab.double(np.array(A).tolist())
    b = matlab.double(np.array(b).tolist())
    Aeq = matlab.double(np.array(Aeq).tolist())
    beq = matlab.double(np.array(beq).tolist())

    # run fmincon
    xopt, ll, exitflag = eng.optimize(function, x0, A, b,
                                      Aeq, beq, lb, ub, nonlcon, options,
                                      providegradients, optional_args,
                                      nargout=3)

    try:
        return np.array(xopt[0]), ll, exitflag
    except TypeError:
        return xopt, ll, exitflag
