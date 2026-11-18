import joblib

model = joblib.load("online-ml-api/model.pkl")

print(type(model))
print("Numero de features:", model.n_features_in_)

if hasattr(model, "feature_names_in_"):
    print(model.feature_names_in_)
else:
    print("No tiene feature_names_in_")