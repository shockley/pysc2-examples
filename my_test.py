import sys
import os
import datetime

from absl import flags
from baselines import deepq
from pysc2.env import sc2_env
from pysc2.lib import actions

from defeat_zerglings import dqfd
from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

step_mul = 1
steps = 2000

FLAGS = flags.FLAGS
start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
flags.DEFINE_string("log", "tensorboard", "logging type(stdout, tensorboard)")
flags.DEFINE_string("algorithm", "deepq", "RL algorithm to use.")
flags.DEFINE_integer("timesteps", 2000000, "Steps to train")
flags.DEFINE_float("exploration_fraction", 0.5, "Exploration Fraction")
flags.DEFINE_boolean("prioritized", True, "prioritized_replay")
flags.DEFINE_boolean("dueling", True, "dueling")
flags.DEFINE_float("lr", 0.001, "Learning rate")

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

max_mean_reward = 0
last_filename = ""

start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")

def main():
  FLAGS(sys.argv)

  logdir = "tensorboard"
  if(FLAGS.algorithm == "deepq"):
    logdir = "tensorboard/zergling/%s/%s_%s_prio%s_duel%s_lr%s/%s" % (
      FLAGS.algorithm,
      FLAGS.timesteps,
      FLAGS.exploration_fraction,
      FLAGS.prioritized,
      FLAGS.dueling,
      FLAGS.lr,
      start_time
    )
  elif(FLAGS.algorithm == "acktr"):
    logdir = "tensorboard/zergling/%s/%s_num%s_lr%s/%s" % (
      FLAGS.algorithm,
      FLAGS.timesteps,
      FLAGS.num_cpu,
      FLAGS.lr,
      start_time
    )

  if(FLAGS.log == "tensorboard"):
    Logger.DEFAULT \
      = Logger.CURRENT \
      = Logger(dir=None,
               output_formats=[TensorBoardOutputFormat(logdir)])

  elif(FLAGS.log == "stdout"):
    Logger.DEFAULT \
      = Logger.CURRENT \
      = Logger(dir=None,
               output_formats=[HumanOutputFormat(sys.stdout)])

  with sc2_env.SC2Env(
      map_name="DefeatZerglingsAndBanelings",
      step_mul=step_mul,
      visualize=True,
      agent_interface_format=sc2_env.AgentInterfaceFormat(feature_dimensions=sc2_env.Dimensions(screen=32, minimap=32)),
      game_steps_per_episode=steps * step_mul) as env:

    print(env.observation_spec())
    screen_dim = env.observation_spec()[0]['feature_screen'][1:3]
    print(screen_dim)


if __name__ == '__main__':
  main()
