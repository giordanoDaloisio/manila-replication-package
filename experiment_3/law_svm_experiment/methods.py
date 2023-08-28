from enum import Enum

class FairnessMethods(Enum):
  RW = 'rw'
  DIR = 'dir'
  DEMV = 'demv'
  EG = 'eg'
  GRID = 'grid'
  AD = 'adv_deb'
  GERRY = 'gerry_fair'
  META = 'meta_fair'
  PREJ = 'prej'
  CAL_EO = 'cal_eo'
  REJ = 'rej_opt'
  NO_ONE = 'no_one'