def preprocess_digits(dataset):
    n_samples=len(dataset.images)
    data=dataset.images.reshape((n_samples,-1))
    label=dataset.target
    return data,label



def train_dev_test_split(data,label,train_frac,dev_frac):
    dev_test_frac=1-train_frac
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(data ,digits.target, test_size=dev_test_frac, shuffle=True,random_state=42)
    X_test, X_dev, y_test, y_dev = train_test_split(X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True,random_state=42)
    return X_train,y_train,X_dev,y_dev,X_test,y_test


