import os
import pandas as pd
import mlflow
import mlflow.sklearn
from pycaret.classification import load_model, predict_model
from sklearn.metrics import log_loss, f1_score

# Caminhos
base_dir = "/Users/flaviagerhardt/PB_Engenharia_ML"
raw_file = os.path.join(base_dir, "data", "raw", "dataset_kobe_prod.parquet")
output_file = os.path.join(base_dir, "data", "processed", "predicoes_producao.parquet")
modelo_path = os.path.join(base_dir, "modelo_kobe_final")

# Colunas utilizadas
colunas_modelo = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']


# Carregar modelo
modelo = load_model(modelo_path)

# Carregar base de produção
df_prod = pd.read_parquet(raw_file)

# Filtrar colunas e remover nulos
df_input = df_prod[colunas_modelo].dropna()


# Iniciar rodada no MLflow
with mlflow.start_run(run_name="PipelineAplicacao"):

    # Gerar previsões
    df_pred = predict_model(modelo, data=df_input)

    # Juntar resultados
    df_resultado = df_input.copy()
    df_resultado["prediction"] = df_pred["prediction_label"]
    df_resultado["score"] = df_pred["prediction_score"]

    # Verificar se rótulo real está disponível na base de produção
    # Verificar se rótulo real está disponível na base de produção
    if "shot_made_flag" in df_prod.columns:
        y_true = df_prod.loc[df_input.index, "shot_made_flag"]
        y_pred = df_pred["prediction_label"]
        y_score = df_pred["prediction_score"]

        # Garantir que os dados estão limpos e numéricos
        y_true = pd.to_numeric(y_true, errors="coerce")
        y_score = pd.to_numeric(y_score, errors="coerce")

        # Filtrar valores válidos (sem NaN ou infinitos)
        valid_idx = y_true.notnull() & y_score.notnull()
        y_true = y_true[valid_idx]
        y_score = y_score[valid_idx]
        y_pred = y_pred[valid_idx]

        # Calcular e logar as métricas
        logloss = log_loss(y_true, y_score)
        f1 = f1_score(y_true, y_pred)

        mlflow.log_metric("log_loss_producao", logloss)
        mlflow.log_metric("f1_score_producao", f1)

        print(f"📊 log_loss: {logloss:.4f} | f1_score: {f1:.4f}")


    # Salvar arquivo com predições
    df_resultado.to_parquet(output_file, index=False)
    mlflow.log_artifact(output_file)

    print(f"✅ Predições salvas em: {output_file}")
