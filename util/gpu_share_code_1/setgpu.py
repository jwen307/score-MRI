from gpu_share.gpu_share_helper import GpuShareHelper
import argparse

# This illustrates if you were to declare your usage for just one script (i.e. training)
def declare_usage_in_script():
    # Initial script setup
    gpu_share = GpuShareHelper()
    gpu_share.setup()

    try:
        # training function, testing function, etc.
        # You could make this a 'super script' by doing something like:

        # train()
        # test()

        # This way you can just let everything run uninterrupted.

        # By putting the cleanup in here, your current_usage.json file will not
        # be deleted and your sheet/calendar updates won't be removed if there is a crash.
        gpu_share.cleanup()
    except Exception as e:
        print(e)

########################################################################
# Create a new python file that could look something like this:
from gpu_share import gpu_share_helper
# OTHER IMPORTS

def set_usage():
    gpu_share = GpuShareHelper()
    gpu_share.setup()


def unset_usage():
    # Recalling setup here will load current_usage.json, allowing you to remove the data
    gpu_share = GpuShareHelper()
    gpu_share.setup()
    gpu_share.cleanup()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--set',
        action='store_true',
        help='Set the GPU (Default is to clear GPU usage)'
        )
    
    args = parser.parse_args()
    
    # Setup args for whether or not you are setting or unsetting your usage
    if args.set:
        set_usage()
    else:
        unset_usage()

