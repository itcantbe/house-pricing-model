from utils.config import FRAC_VAL,STATE_VAL

def split_data(df):
    random_input  = df.sample(frac=FRAC_VAL, random_state=STATE_VAL)
    random_test = df[~df.index.isin(random_input.index)]
    random_input_target = random_input[["median_house_value"]].copy()
    random_input_features = random_input.drop(columns=["median_house_value"]).copy()
    random_test_target = random_test[["median_house_value"]].copy()
    random_test_features = random_test.drop(columns=["median_house_value"]).copy()
    return random_input_target,random_input_features,random_test_target,random_test_features

def skew_in_data(random_input_features,random_input_target):
    return random_input_features.skew(), random_input_target.skew()

