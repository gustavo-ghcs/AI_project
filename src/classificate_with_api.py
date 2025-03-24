from contextlib import asynccontextmanager
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
import numpy as np
import pandas as pd  # type: ignore
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler  # type: ignore
import shutil
import uuid
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Configuração do logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

color_map = {
    1: "#0009DE",  # Terreno (ground)
    2: "#7EFF0C",  # Vegetação (vegetation)
    3: "#FB0304"   # Edifícios (buildings)
}

# Diretórios dos arquivos
models_folder = Path("./models/")
download_data_folder = Path("./download_data/")
output_data_folder = Path("./classified_data/")
output_images_folder = Path("./images/")

rf_classifier = None


def carregar_modelo(model_path: Path):
    logging.info("Carregando modelo treinado")
    with model_path.open("rb") as f:
        return pickle.load(f)


def carregar_nuvem_pontos(dataset_path: Path):
    logging.info("Carregando nova nuvem de pontos")
    new_pcd = pd.read_csv(dataset_path, delimiter=" ")
    new_pcd.dropna(inplace=True)
    return new_pcd


def selecionar_features(new_pcd: pd.DataFrame):
    logging.info("Selecionando features")
    features_columns = [
        "Z",
        "R",
        "G",
        "B",
        "omnivariance_2",
        "normal_cr_2",
        "NumberOfReturns",
        "planarity_2",
        "omnivariance_1",
        "verticality_1",
    ]
    return new_pcd[features_columns]


def normalizar_features(new_features: pd.DataFrame):
    logging.info("Normalizando features")
    scaler = MinMaxScaler()
    return scaler.fit_transform(new_features)


def fazer_previsoes(model, new_features_scaled):
    logging.info("Fazendo previsões")
    return model.predict(new_features_scaled)


def gerar_imagens(new_pcd: pd.DataFrame, predictions, image_path1: Path):
    logging.info("Gerando imagens da Nuvem de Pontos Classificada")
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = np.array([color_map[class_id] for class_id in predictions])
    ax.scatter(
        new_pcd["X"], new_pcd["Y"], c=colors, s=0.05
    )
    ax.set_title("3D Point Cloud Predictions")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.savefig(image_path1, dpi=300)

    # fig, ax = plt.subplots(figsize=(6, 6))
    # scatter = ax.scatter(new_pcd['X'], new_pcd['Z'], c=predictions, cmap="plasma", s=0.05)
    # ax.set_title('3D Point Cloud Predictions - Alternative View')
    # ax.set_xlabel("X")
    # ax.set_ylabel("Z")
    # plt.savefig(image_path2, dpi=300)

    logging.info(f"Imagens salvas em: {image_path1}")


# Called before API start
async def startup():
    global rf_classifier

    # Criar diretórios, se não existirem
    download_data_folder.mkdir(parents=True, exist_ok=True)
    output_data_folder.mkdir(parents=True, exist_ok=True)
    models_folder.mkdir(parents=True, exist_ok=True)
    output_images_folder.mkdir(parents=True, exist_ok=True)

    model_path = models_folder / "model_trained.pkl"
    rf_classifier = carregar_modelo(model_path)


# Called before shutdown
async def shutdown(): ...


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    yield
    await shutdown()


app = FastAPI(title="ProjetoAI API", lifespan=lifespan)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    if rf_classifier is None:
        return JSONResponse(content={"error": "Modelo não carregado"}, status_code=500)

    unique_id = str(uuid.uuid4())

    logging.info(f"Processando arquivo: {file.filename} as {unique_id}")

    dataset_path = download_data_folder / f"{unique_id}.xyz"
    output_image1 = output_images_folder / f"{unique_id}_view1.jpg"

    # with file.file as buffer:
    with dataset_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


    new_pcd = carregar_nuvem_pontos(dataset_path)

    new_features = selecionar_features(new_pcd)
    new_features_scaled = normalizar_features(new_features)

    predictions = fazer_previsoes(rf_classifier, new_features_scaled)
    new_pcd["Classification"] = predictions

    logging.info(
        f"Salvando nuvem classificada em: {output_data_folder / f'{unique_id}.xyz'}"
    )
    new_pcd.to_csv(
        output_data_folder / f"{unique_id}.xyz", sep=" ", index=False, header=True
    )

    gerar_imagens(new_pcd, predictions, output_image1)

    # result_json = new_pcd[['X', 'Y', 'Z', 'Classification']].to_dict(orient='records')

    return JSONResponse(
        content={
            "message": "Processamento concluído com sucesso",
            "classified_data": f"{unique_id}.xyz",
            "image1": f"{unique_id}_view1.jpg",
        }
    )


@app.get("/image/{filename}")
async def download_image(filename: str):
    file_path = output_images_folder / filename
    if file_path.exists():
        return FileResponse(file_path, media_type="image/jpeg", filename=filename)
    return JSONResponse(content={"error": "Arquivo não encontrado"}, status_code=404)


@app.get("/data/{filename}")
async def download_data(filename: str):
    file_path = output_data_folder / filename
    if file_path.exists():
        return FileResponse(file_path, media_type="text/plain", filename=filename)
    return JSONResponse(content={"error": "Arquivo não encontrado"}, status_code=404)
