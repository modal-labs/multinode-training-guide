
right now i want to to write miles_v2. you need to completely ignore the previous implementation of miles in this repo(i deleted it, you can see in git, do not read anything about it). You need to check the styles in slime instead, as that is the specific style I want. 

For reference, you should go to the source /home/ec2-user/nan_wonderland/miles for more information about miles and for context, miles is a fork of slime, so the styles are very similar.

You should definitely check the entire /home/ec2-user/multinode-training-guide/slime and /home/ec2-user/nan_wonderland/slime repo—specifically to see the pattern of what we inject and how to include miles.

the docker we will use for miles is `radixark/miles:dev-202604101227` and it is same as `radixark/miles:latest` if you run `docker images`.

Specifically, you need to check:
1. What our abstraction is for the separate miles, especially when there are local miles where our abstraction is going to be used.
2. What object you need to add to the support miles, particularly where local miles require that object.

since we are designing for miles launcher for modal, you should get yourself familiar with modal, https://modal.com/llms.txt is a good starting point.

To verify if you finish it or not, you should create a corresponding config of `/home/ec2-user/nan_wonderland/miles/examples/lora/run-qwen3-4b-megatron-lora-result.sh` and run it with modal and check if it can finish two training steps. YOU DO NOT NEED ANYTHING FROM `/home/ec2-user/multinode-training-guide/miles`, miles docker should have everything you need to run lora stuff for qwen3-4b. so you should completely ditch and ignore everything in `/home/ec2-user/multinode-training-guide/miles`

you should create branch based on `nan/slime-refactor` here right now in multinode-training-guide