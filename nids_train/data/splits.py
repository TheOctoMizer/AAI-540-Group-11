from sklearn.model_selection import train_test_split

def split_benign_data(X, random_state=42):
    X_temp, X_prod = train_test_split(X, test_size=0.4, random_state=random_state)
    X_train, X_temp = train_test_split(X_temp, test_size=0.333, random_state=random_state)
    X_test, X_val = train_test_split(X_temp, test_size=0.5, random_state=random_state)
    
    return {
        "train": X_train,
        "val": X_val,
        "test": X_test,
        "production": X_prod
    }