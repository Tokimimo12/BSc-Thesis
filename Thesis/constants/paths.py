from typing import Optional

# path to the SDK folder
SDK_PATH: Optional[str] = '/home/viki/Documents/Thesis/TutorialCMU/CMU-MultimodalSDK'
print("SDK_PATH:", SDK_PATH)

# path to the folder where you want to store data
DATA_PATH: Optional[str] = '/home/viki/Documents/Thesis/TutorialCMU/data'

# path to a pretrained word embedding file
WORD_EMB_PATH: Optional[str] = '/home/viki/Documents/Thesis/TutorialCMU/data/http__immortal.multicomp.cs.cmu.edu/CMU-MOSI/language/CMU_MOSI_TimestampedWordVectors.csd'

# path to loaded word embedding matrix and corresponding word2id mapping
CACHE_PATH: Optional[str] = None
# CACHE_PATH: Optional[str] = 'C:\\Users\\Viki\\Documents\\Thesis\\tryout3\\data\\embedding_and_mapping.pt'

