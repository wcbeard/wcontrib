def check_gpu():
    """
    From Google's quickstart
    """
    # gpu_info = !nvidia-smi
    gpu_info = get_ipython().getoutput("nvidia-smi")
    gpu_info = "\n".join(gpu_info)
    if gpu_info.find("failed") >= 0:
        print(
            'Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, '
        )
        print("and then re-execute this cell.")
    else:
        print(gpu_info)