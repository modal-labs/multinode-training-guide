# To run the training, we can use the following command:
# ```bash
# modal run learn_math.py --mode=train --trainer-script=trainer_script_grpo.py --config-file=config_grpo.yaml
# ```
# To run the inference with a custom prompt, we can use the following command:
# ```bash
# modal run learn_math.py --mode=inference --prompt "Find the value of x that satisfies the equation: 2x + 5 = 17"
# ```
# To run the inference with a custom prompt from a file, we can use the following command:
# ```bash
# modal run learn_math.py --mode=inference --prompt-file "prompt.txt"