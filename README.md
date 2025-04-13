# FSDP Implementation

This is a self-made FSDP FULL_SHARD (Level 3) algorithm implementation that I made as a homework for YSDA Efficient DL course. It can be used as an alternative to PyTorch FSDP, giving the same outputs/gradients and memory/speed footprints. 

The repo also contains a showcase of usage - a simple Llama training. Follow these steps to run the training:


1. Run ```bash setup.sh``` to install all the needed imports
2. Run ```torchrun train.py``` with the following flags:
    - ```--nproc_per_node``` specifies the number of Torch processes
    - ```--hw_fsdp``` specifies the FSDP implementation (DIY/Torch one)
    - ```--model_name``` specifies the model name from HF library
    - ```--batch_size``` specifies the batch size
    - ```--seq_len``` specifies the maximum sequence length
    - ```--training_steps``` specifies the number of training sets
    - ```--warmup_steps``` specifies the number of warmup steps 
    - ```--param_dtype``` specifies the precision of parameters for the forward/backward pass
    - ```--reduce_dtype``` specifies the precision of parameters in reduce-scatter syncronizing operation during backward pass
    - ```--reshard_after_forward``` specifies if resharding after forward pass is needed
    - ```--lr``` specifies the learning rate


Huge thanks to @lovanto for creating this assignment and the template given!
