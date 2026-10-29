# SuperMarket Sales Prediction & Analytics Intelligence System


from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError, WriteError
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "supermarket_db"
COLLECTION_NAME = "sales"
DATA_FOLDER = Path(r"C:\Users\adars\OneDrive\Desktop\Supermarket")
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
RANDOM_STATE = 42
SHOW_PLOTS = False  # set True to also pop up the charts on screen

sns.set_theme(style="whitegrid")


def banner(title: str) -> None:
    """Print a section heading so the console output is easy to present."""
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def find_csv() -> Path:
    """Locate the dataset: user's folder first, then the script's own folder."""
    for folder in (DATA_FOLDER, Path(__file__).resolve().parent):
        if folder.exists():
            exact = folder / "SuperMarket Analysis.csv"
            if exact.exists():
                return exact
            matches = sorted(folder.glob("*upermarket*nalysis*.csv")) + sorted(
                folder.glob("*upermarket*.csv")
            )
            if matches:
                return matches[0]
    raise FileNotFoundError(
        "Could not find the SuperMarket CSV. Put it in "
        f"{DATA_FOLDER} or next to main.py."
    )


# --------------------------------------------------------------------------- #
# 1. DATABASE SETUP + VALIDATION RULES
# --------------------------------------------------------------------------- #
def get_collection(client: MongoClient) -> Collection:
    """Create (or re-create) the collection with a JSON Schema validator.

    Why validation? MongoDB is schema-flexible, so without rules a typo such as
    rating = "ten" would be stored silently. The validator rejects bad documents
    at the database level, before they can corrupt analysis.
    """
    db = client[DB_NAME]
    if COLLECTION_NAME in db.list_collection_names():
        db.drop_collection(COLLECTION_NAME)  # makes the script re-runnable

    num = ["double", "int", "long"]
    schema = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [
                "invoice_id", "store", "customer", "product",
                "financials", "payment_method", "purchase_datetime", "rating",
            ],
            "properties": {
                "invoice_id": {
                    "bsonType": "string",
                    "pattern": r"^\d{3}-\d{2}-\d{4}$",
                    "description": "format 123-45-6789, required",
                },
                "store": {
                    "bsonType": "object",
                    "required": ["branch", "city"],
                    "properties": {
                        "branch": {"enum": ["Alex", "Giza", "Cairo"]},
                        "city": {"enum": ["Yangon", "Mandalay", "Naypyitaw"]},
                    },
                },
                "customer": {
                    "bsonType": "object",
                    "required": ["type", "gender"],
                    "properties": {
                        "type": {"enum": ["Member", "Normal"]},
                        "gender": {"enum": ["Male", "Female"]},
                    },
                },
                "product": {
                    "bsonType": "object",
                    "required": ["line", "unit_price", "quantity"],
                    "properties": {
                        "line": {"bsonType": "string"},
                        "unit_price": {"bsonType": num, "minimum": 0,
                                       "description": "must be >= 0"},
                        "quantity": {"bsonType": ["int", "long"], "minimum": 1,
                                     "maximum": 10},
                    },
                },
                "financials": {
                    "bsonType": "object",
                    "required": ["sales", "tax", "cogs", "gross_income"],
                    "properties": {
                        "sales": {"bsonType": num, "minimum": 0},
                        "tax": {"bsonType": num, "minimum": 0},
                        "cogs": {"bsonType": num, "minimum": 0},
                        "gross_income": {"bsonType": num, "minimum": 0},
                    },
                },
                "payment_method": {"enum": ["Cash", "Ewallet", "Credit card"]},
                "purchase_datetime": {"bsonType": "date"},
                "rating": {"bsonType": num, "minimum": 1, "maximum": 10,
                           "description": "customer rating between 1 and 10"},
            },
        }
    }
    collection = db.create_collection(
        COLLECTION_NAME, validator=schema, validationAction="error",
        validationLevel="strict",
    )
    collection.create_index("invoice_id", unique=True)  # no duplicate invoices
    print(f"Database '{DB_NAME}' ready, collection '{COLLECTION_NAME}' created "
          "with JSON Schema validation and a unique index on invoice_id.")
    return collection


# --------------------------------------------------------------------------- #
# 2. DATA LOADING (CREATE): CSV -> documents
# --------------------------------------------------------------------------- #
def row_to_document(row: pd.Series) -> dict[str, Any]:
    """Convert one CSV row into a nested MongoDB document.

    Numpy types are cast to native Python types because PyMongo (BSON)
    cannot encode numpy.int64 / numpy.float64.
    """
    return {
        "invoice_id": str(row["Invoice ID"]),
        "store": {"branch": str(row["Branch"]), "city": str(row["City"])},
        "customer": {"type": str(row["Customer type"]),
                     "gender": str(row["Gender"])},
        "product": {
            "line": str(row["Product line"]),
            "unit_price": float(row["Unit price"]),
            "quantity": int(row["Quantity"]),
        },
        "financials": {
            "sales": float(row["Sales"]),
            "tax": float(row["Tax 5%"]),
            "cogs": float(row["cogs"]),
            "gross_margin_pct": float(row["gross margin percentage"]),
            "gross_income": float(row["gross income"]),
        },
        "payment_method": str(row["Payment"]),
        "purchase_datetime": row["purchase_datetime"].to_pydatetime(),
        "rating": float(row["Rating"]),
    }


def load_csv_to_mongo(collection: Collection) -> None:
    """Read the CSV, clean it, and bulk-insert it into MongoDB."""
    csv_path = find_csv()
    df = pd.read_csv(csv_path, encoding="utf-8-sig")  # utf-8-sig strips the BOM
    print(f"Loaded '{csv_path.name}': {df.shape[0]} rows x {df.shape[1]} columns")

    df = df.drop_duplicates(subset="Invoice ID")
    df = df.dropna()
    df["purchase_datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%m/%d/%Y %I:%M:%S %p"
    )

    documents = [row_to_document(r) for _, r in df.iterrows()]
    result = collection.insert_many(documents)
    print(f"Inserted {len(result.inserted_ids)} documents into MongoDB.")
    print("\nSample document stored in MongoDB:")
    print(json.dumps(collection.find_one({}, {"_id": 0}), indent=2, default=str))

    # Demonstrate that validation really works: this bad document must fail.
    bad_doc = documents[0].copy()
    bad_doc["invoice_id"] = "000-00-0000"
    bad_doc["rating"] = 25.0  # violates maximum: 10
    try:
        collection.insert_one(bad_doc)
        print("WARNING: invalid document was accepted (validation not active).")
    except WriteError as err:
        print("\nValidation test: invalid document (rating = 25) was REJECTED "
              f"-> error code {err.code} (DocumentValidationFailure).")


# --------------------------------------------------------------------------- #
# 3. CRUD OPERATIONS
# --------------------------------------------------------------------------- #
def crud_demo(collection: Collection) -> None:
    """One Create, Read, Update and Delete on a dedicated test invoice."""
    test_id = "999-99-9999"

    # CREATE
    new_doc = {
        "invoice_id": test_id,
        "store": {"branch": "Alex", "city": "Yangon"},
        "customer": {"type": "Member", "gender": "Female"},
        "product": {"line": "Health and beauty", "unit_price": 50.0, "quantity": 4},
        "financials": {"sales": 210.0, "tax": 10.0, "cogs": 200.0,
                       "gross_margin_pct": 4.761904762, "gross_income": 10.0},
        "payment_method": "Cash",
        "purchase_datetime": datetime(2019, 3, 30, 14, 30),
        "rating": 7.5,
    }
    collection.insert_one(new_doc)
    print(f"[CREATE] Inserted invoice {test_id}")

    # READ
    found = collection.find_one({"invoice_id": test_id}, {"_id": 0})
    print("[READ]   find_one ->", json.dumps(found, default=str))
    top3 = list(collection.find(
        {"financials.sales": {"$gt": 1000}},
        {"_id": 0, "invoice_id": 1, "financials.sales": 1, "store.city": 1},
    ).sort("financials.sales", -1).limit(3))
    print("[READ]   Top 3 invoices with sales > 1000:")
    for doc in top3:
        print("         ", doc)

    # UPDATE
    res = collection.update_one(
        {"invoice_id": test_id},
        {"$set": {"rating": 9.0, "payment_method": "Ewallet"}},
    )
    updated = collection.find_one({"invoice_id": test_id}, {"_id": 0, "rating": 1,
                                                          "payment_method": 1})
    print(f"[UPDATE] matched={res.matched_count}, modified={res.modified_count} "
          f"-> {updated}")

    # DELETE
    res = collection.delete_one({"invoice_id": test_id})
    print(f"[DELETE] deleted_count={res.deleted_count}; documents now = "
          f"{collection.count_documents({})}")


# --------------------------------------------------------------------------- #
# 4. AGGREGATION PIPELINES
# --------------------------------------------------------------------------- #
def run_aggregations(collection: Collection) -> dict[str, pd.DataFrame]:
    """Run four aggregation pipelines and return the results as DataFrames."""
    pipelines: dict[str, list[dict]] = {
        "Total & average sales per city/branch": [
            {"$group": {
                "_id": {"city": "$store.city", "branch": "$store.branch"},
                "total_sales": {"$sum": "$financials.sales"},
                "avg_sale": {"$avg": "$financials.sales"},
                "invoices": {"$sum": 1},
            }},
            {"$project": {"_id": 0, "city": "$_id.city", "branch": "$_id.branch",
                          "total_sales": "$total_sales",
                          "avg_sale": "$avg_sale",
                          "invoices": 1}},
            {"$sort": {"total_sales": -1}},
        ],
        "Average rating & total sales per product line": [
            {"$group": {
                "_id": "$product.line",
                "avg_rating": {"$avg": "$rating"},
                "total_sales": {"$sum": "$financials.sales"},
            }},
            {"$project": {"_id": 0, "product_line": "$_id",
                          "avg_rating": "$avg_rating",
                          "total_sales": "$total_sales"}},
            {"$sort": {"total_sales": -1}},
        ],
        "Sales by customer type and payment method": [
            {"$group": {
                "_id": {"customer_type": "$customer.type",
                        "payment": "$payment_method"},
                "avg_sale": {"$avg": "$financials.sales"},
                "invoices": {"$sum": 1},
            }},
            {"$project": {"_id": 0, "customer_type": "$_id.customer_type",
                          "payment": "$_id.payment",
                          "avg_sale": "$avg_sale",
                          "invoices": 1}},
            {"$sort": {"customer_type": 1, "avg_sale": -1}},
        ],
        "Monthly sales trend": [
            {"$group": {
                "_id": {"$month": "$purchase_datetime"},
                "total_sales": {"$sum": "$financials.sales"},
                "invoices": {"$sum": 1},
            }},
            {"$project": {"_id": 0, "month": "$_id",
                          "total_sales": "$total_sales",
                          "invoices": 1}},
            {"$sort": {"month": 1}},
        ],
    }
    results: dict[str, pd.DataFrame] = {}
    for title, pipeline in pipelines.items():
        df = pd.DataFrame(list(collection.aggregate(pipeline))).round(2)
        results[title] = df
        print(f"\n--- {title} ---")
        print(df.to_string(index=False))
    return results


# --------------------------------------------------------------------------- #
# 5. DATA PREPROCESSING
# --------------------------------------------------------------------------- #
CATEGORICAL = ["city", "customer_type", "gender", "product_line", "payment_method"]
NUMERIC = ["unit_price", "quantity", "hour", "month", "weekday"]


def load_dataframe(collection: Collection) -> pd.DataFrame:
    """Fetch documents from MongoDB and flatten the nested structure."""
    docs = list(collection.find({}, {"_id": 0}))
    flat = pd.json_normalize(docs, sep="_")
    flat = flat.rename(columns={
        "store_city": "city", "store_branch": "branch",
        "customer_type": "customer_type", "customer_gender": "gender",
        "product_line": "product_line", "product_unit_price": "unit_price",
        "product_quantity": "quantity", "financials_sales": "sales",
    })
    flat["purchase_datetime"] = pd.to_datetime(flat["purchase_datetime"])
    flat["hour"] = flat["purchase_datetime"].dt.hour
    flat["month"] = flat["purchase_datetime"].dt.month
    flat["weekday"] = flat["purchase_datetime"].dt.weekday
    return flat


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode categorical columns, pass numeric columns through."""
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
         ("num", "passthrough", NUMERIC)]
    )


def preprocess_report(df: pd.DataFrame) -> None:
    """Print the data-quality checks that justify our feature choices."""
    print(f"Shape retrieved from MongoDB: {df.shape}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"Duplicate invoices: {int(df['invoice_id'].duplicated().sum())}")

    # Branch and City carry identical information in this dataset.
    mapping = df.groupby("branch")["city"].nunique()
    print("Cities per branch (1 = perfect one-to-one mapping):",
          mapping.to_dict())
    print("-> 'branch' is dropped from the features because 'city' already "
          "contains the same information.")

    # Sales is a formula of unit price and quantity.
    ratio = (df["sales"] / (df["unit_price"] * df["quantity"])).round(4).unique()
    print(f"Sales / (unit_price x quantity) = {ratio}  (i.e. price x qty x 1.05 tax)")
    print("-> tax, cogs and gross_income are EXCLUDED from features: they are "
          "computed from Sales (data leakage).")


# --------------------------------------------------------------------------- #
# 6. MACHINE LEARNING
# --------------------------------------------------------------------------- #
def regression_experiment(df: pd.DataFrame) -> dict[str, Any]:
    """Predict the Sales amount of an invoice (Linear Regression vs Random Forest)."""
    X, y = df[CATEGORICAL + NUMERIC], df["sales"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train size = {len(X_train)}, Test size = {len(X_test)} (80/20 split)")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    }
    out: dict[str, Any] = {"models": {}}
    for name, model in models.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        cv_r2 = cross_val_score(pipe, X, y, cv=5, scoring="r2")
        metrics = {
            "R2": r2_score(y_test, pred),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "CV_R2_mean": float(cv_r2.mean()),
            "CV_R2_std": float(cv_r2.std()),
        }
        out["models"][name] = metrics
        out[name] = (pipe, pred)
        print(f"\n{name}: R2 = {metrics['R2']:.4f} | MAE = {metrics['MAE']:.2f} | "
              f"RMSE = {metrics['RMSE']:.2f} | 5-fold CV R2 = "
              f"{metrics['CV_R2_mean']:.4f} +/- {metrics['CV_R2_std']:.4f}")

    out["y_test"] = y_test
    out["baseline_mae"] = float(mean_absolute_error(
        y_test, np.full(len(y_test), y_train.mean())))
    print(f"\nBaseline (always predict the mean) MAE = {out['baseline_mae']:.2f}")

    # Sample predictions to show in the demo
    best = max(out["models"], key=lambda k: out["models"][k]["R2"])
    out["best_regressor"] = best
    sample = X_test.head(5).copy()
    sample["actual_sales"] = y_test.head(5).round(2)
    sample["predicted_sales"] = out[best][1][:5].round(2)
    print(f"\nSample predictions ({best}):")
    print(sample[["city", "product_line", "unit_price", "quantity",
                  "actual_sales", "predicted_sales"]].to_string(index=False))
    return out


def classification_experiment(df: pd.DataFrame) -> dict[str, Any]:
    """Classify an invoice as High-value (sales >= median) or Low-value."""
    threshold = float(df["sales"].median())
    y = (df["sales"] >= threshold).astype(int)
    X = df[CATEGORICAL + NUMERIC]
    print(f"High-value invoice = Sales >= median ({threshold:.2f}); "
          f"class balance: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    prep_scaled = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
         ("num", StandardScaler(), NUMERIC)]
    )
    models = {
        "Logistic Regression": Pipeline([
            ("prep", prep_scaled),
            ("model", LogisticRegression(max_iter=1000))]),
        "Random Forest Classifier": Pipeline([
            ("prep", build_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=200,
                                             random_state=RANDOM_STATE, n_jobs=-1))]),
    }
    out: dict[str, Any] = {"models": {}, "threshold": threshold}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
        }
        out["models"][name] = metrics
        out[name] = (pipe, pred)
        print(f"\n{name}: " + " | ".join(f"{k} = {v:.4f}" for k, v in metrics.items()))
        print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, pred))
    out["y_test"] = y_test
    out["best_classifier"] = max(out["models"], key=lambda k: out["models"][k]["F1"])
    return out


# --------------------------------------------------------------------------- #
# 7. VISUALISATION
# --------------------------------------------------------------------------- #
def make_charts(df: pd.DataFrame, agg: dict[str, pd.DataFrame],
                reg: dict[str, Any], clf: dict[str, Any]) -> None:
    """Create and save five charts into ./output."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Total sales by city
    city = agg["Total & average sales per city/branch"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=city, x="city", y="total_sales", hue="city",
                palette="viridis", legend=False, ax=ax)
    ax.set_title("Total Sales by City (from MongoDB aggregation)")
    ax.set_xlabel("City"); ax.set_ylabel("Total sales")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():,.0f}", (p.get_x() + p.get_width() / 2,
                    p.get_height()), ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "1_sales_by_city.png", dpi=150)

    # 2. Impact of product line on sales
    fig, ax = plt.subplots(figsize=(9, 5))
    order = df.groupby("product_line")["sales"].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x="sales", y="product_line", order=order, hue="product_line",
                palette="mako", legend=False, errorbar=None, ax=ax)
    ax.set_title("Impact of Product Line on Average Sales per Invoice")
    ax.set_xlabel("Average sales"); ax.set_ylabel("")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "2_product_line_vs_sales.png", dpi=150)

    # 3. Actual vs predicted (regression)
    best = reg["best_regressor"]
    y_test, pred = reg["y_test"], reg[best][1]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, pred, alpha=0.6, edgecolor="k", linewidth=0.3)
    lim = [0, max(y_test.max(), pred.max()) * 1.05]
    ax.plot(lim, lim, "r--", label="Perfect prediction")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"Actual vs Predicted Sales ({best})\n"
                 f"R² = {reg['models'][best]['R2']:.3f}, "
                 f"MAE = {reg['models'][best]['MAE']:.2f}")
    ax.set_xlabel("Actual sales"); ax.set_ylabel("Predicted sales"); ax.legend()
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "3_actual_vs_predicted.png", dpi=150)

    # 4. Feature importance of the Random Forest regressor
    rf_pipe = reg["Random Forest Regressor"][0]
    names = rf_pipe.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    imp = pd.Series(rf_pipe.named_steps["model"].feature_importances_, index=names)
    imp = imp.sort_values(ascending=False).head(8)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    imp.plot.barh(color="teal", ax=ax)
    ax.set_title("Top Features Driving Sales (Random Forest)")
    ax.set_xlabel("Importance")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "4_feature_importance.png", dpi=150)

    # 5. Confusion matrix of the best classifier
    bc = clf["best_classifier"]
    cm = confusion_matrix(clf["y_test"], clf[bc][1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Low", "High"], yticklabels=["Low", "High"], ax=ax)
    ax.set_title(f"Confusion Matrix ({bc})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "5_confusion_matrix.png", dpi=150)

    print(f"Saved 5 charts to: {OUTPUT_DIR}")
    if SHOW_PLOTS:
        plt.show()
    plt.close("all")


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    banner("STEP 1: DATABASE SETUP & VALIDATION RULES")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=4000)
        client.admin.command("ping")
    except ServerSelectionTimeoutError:
        raise SystemExit("Cannot reach MongoDB. Start the MongoDB service "
                         f"(expected at {MONGO_URI}) and run again.")
    collection = get_collection(client)

    banner("STEP 2: LOAD CSV INTO MONGODB (CREATE)")
    load_csv_to_mongo(collection)

    banner("STEP 3: CRUD OPERATIONS")
    crud_demo(collection)

    banner("STEP 4: AGGREGATION PIPELINES")
    agg = run_aggregations(collection)

    banner("STEP 5: DATA PREPROCESSING (MongoDB -> Pandas)")
    df = load_dataframe(collection)
    preprocess_report(df)

    banner("STEP 6A: REGRESSION - PREDICT SALES")
    reg = regression_experiment(df)

    banner("STEP 6B: CLASSIFICATION - HIGH vs LOW VALUE INVOICE")
    clf = classification_experiment(df)

    banner("STEP 7: VISUALISATIONS")
    make_charts(df, agg, reg, clf)

    summary = {
        "regression": reg["models"], "baseline_mae": reg["baseline_mae"],
        "classification": clf["models"], "median_threshold": clf["threshold"],
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(summary, indent=2))
    print("\nAll steps completed. Metrics saved to output/results.json")
    client.close()


if __name__ == "__main__":
    try:
        main()
    except PyMongoError as exc:
        raise SystemExit(f"MongoDB error: {exc}")
