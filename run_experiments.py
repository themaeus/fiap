import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ga_optimizer import OtimizadorGA

OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def carregar_dataset():
    import kagglehub
    from pathlib import Path
    path = kagglehub.dataset_download('uciml/breast-cancer-wisconsin-data')
    dataset_path = Path(path)
    csv_files = sorted(dataset_path.rglob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'Nenhum CSV encontrado em {dataset_path}')
    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)
    if 'Unnamed: 32' in df.columns:
        df = df.drop(columns=['Unnamed: 32'])
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    df['target'] = df['diagnosis'].map({'B': 0, 'M': 1})
    return df


def criar_preprocessador(colunas_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, colunas_features)])
    return preprocessor


def avaliar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
    }


def construtor_rf(preprocessor):
    def construir(params):
        n_estimators = int(params.get('n_estimators', 100))
        max_depth = None if params.get('max_depth', 0) == 0 else int(params.get('max_depth'))
        min_samples_split = int(params.get('min_samples_split', 2))
        max_features = params.get('max_features', 'sqrt')
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                    min_samples_split=min_samples_split, max_features=max_features, random_state=42)
        return Pipeline(steps=[('preprocessor', preprocessor), ('model', rf)])
    return construir


def construtor_lr(preprocessor):
    def construir(params):
        C = float(params.get('C', 1.0))
        lr = LogisticRegression(C=C, max_iter=2000, random_state=42)
        return Pipeline(steps=[('preprocessor', preprocessor), ('model', lr)])
    return construir


def main():
    df = carregar_dataset()
    colunas_features = [c for c in df.columns if c not in ['diagnosis', 'target']]
    # manter como DataFrame para que ColumnTransformer aceite nomes de colunas
    X = df[colunas_features]
    y = df['target']

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=1, stratify=y_train_full)

    preprocessor = criar_preprocessador(colunas_features)

    # Baselines
    baseline_lr = Pipeline(steps=[('preprocessor', preprocessor), ('model', LogisticRegression(max_iter=2000, random_state=42))])
    baseline_rf = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestClassifier(n_estimators=300, random_state=42))])

    baseline_lr.fit(X_train_full, y_train_full)
    baseline_rf.fit(X_train_full, y_train_full)

    baseline_results = {
        'LogisticRegression': avaliar_modelo(baseline_lr, X_test, y_test),
        'RandomForest': avaliar_modelo(baseline_rf, X_test, y_test)
    }

    print('Resultados baseline:', baseline_results)

    # Espaços de busca
    rf_space = {
        'n_estimators': {'type': 'int', 'min': 50, 'max': 300},
        'max_depth': {'type': 'int', 'min': 0, 'max': 30},
        'min_samples_split': {'type': 'int', 'min': 2, 'max': 10},
        'max_features': {'type': 'cat', 'choices': ['sqrt', 'log2', None]}
    }

    lr_space = {
        'C': {'type': 'float', 'min': 1e-4, 'max': 100.0}
    }

    # Configurações do GA (3 experimentos)
    ga_configs = [
        {'tam_pop': 20, 'geracoes': 20, 'taxa_mutacao': 0.1},
        {'tam_pop': 40, 'geracoes': 30, 'taxa_mutacao': 0.05},
        {'tam_pop': 60, 'geracoes': 25, 'taxa_mutacao': 0.2},
    ]

    resultados = {'baseline': baseline_results, 'experimentos': []}

    for idx, cfg in enumerate(ga_configs, start=1):
        print(f'Executando experimento GA {idx} com cfg: {cfg}')
        # Random Forest
        rf_builder = construtor_rf(preprocessor)
        rf_ga = OtimizadorGA(construir_modelo=rf_builder, espaco_params=rf_space, tam_populacao=cfg['tam_pop'], geracoes=cfg['geracoes'], taxa_mutacao=cfg['taxa_mutacao'])
        rf_out = rf_ga.fit(X_train, y_train, X_val, y_val, verbose=True)
        rf_eval = avaliar_modelo(rf_out['model'], X_test, y_test)

        # Logistic Regression
        lr_builder = construtor_lr(preprocessor)
        lr_ga = OtimizadorGA(construir_modelo=lr_builder, espaco_params=lr_space, tam_populacao=cfg['tam_pop'], geracoes=cfg['geracoes'], taxa_mutacao=cfg['taxa_mutacao'])
        lr_out = lr_ga.fit(X_train, y_train, X_val, y_val, verbose=True)
        lr_eval = avaliar_modelo(lr_out['model'], X_test, y_test)

        exp = {
            'config': cfg,
            'rf': {'best_params': rf_out['best_params'], 'validation_metrics': rf_out['best_metrics'], 'test_metrics': rf_eval},
            'lr': {'best_params': lr_out['best_params'], 'validation_metrics': lr_out['best_metrics'], 'test_metrics': lr_eval}
        }
        resultados['experimentos'].append(exp)

        out_path = OUTPUT_DIR / f'ga_experimento_{idx}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(exp, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / 'ga_resultados_sumario.json', 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print('Todos os experimentos completos. Resultados salvos em outputs/')


if __name__ == '__main__':
    main()
