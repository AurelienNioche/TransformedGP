function [xopt, fopt, exitflag] = optimize(funcname, x0, A, b, ...
    Aeq, beq, lb, ub, nonlc, opt_struct, gradients, optional_args)

    % set options
    options = optimoptions('fmincon');
    names = fieldnames(opt_struct);
    for i = 1:length(names)
        options = optimoptions(options, names{i}, opt_struct.(names{i}));
    end

    [xopt, fopt, exitflag, output] = fmincon(@obj, x0, A, b, Aeq, beq, lb, ub, nonlc, options);


    function x = fupdate(x)
        eval(['x = py.', funcname, '(x);'])
    end


    % ---------- Objective Function ------------------
    function [x] = obj(x)
        x = double(fupdate(x));
    end
    % -------------------------------------------------

end
