def check_grad(model, verbose=True, debug=False):
    must_break = False
    for k, p in model.named_parameters():
        if (p.grad is not None) and (p.grad != p.grad).any():
            must_break = True
            break

    if must_break:
        if verbose:
            print("grad")
            for k, p in model.named_parameters():
                if p.grad is not None:
                    print(k, p.grad.mean(), p.grad.std(), p.grad.min(), p.grad.max())

        if debug:
            breakpoint()
        else:
            exit(0)
