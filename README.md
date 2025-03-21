# Projeto de Classificação de Nuvens de Pontos 3D

## Descrição do Projeto
Este projeto tem como objetivo classificar elementos em nuvens de pontos 3D utilizando técnicas de Machine Learning, especificamente um modelo baseado em Random Forest. Os dados utilizados incluem informações espaciais e espectrais dos pontos, como coordenadas (X, Y, Z), valores de cor (R, G, B) e atributos derivados da análise da nuvem.

O código está pronto para ser colocado em um cluster para treinamento, caso seja necessário trabalhar com datasets mais pesados.

## Estrutura do Projeto
O projeto segue as seguintes etapas:

1. **Definição do Problema e Coleta de Dados**  
   - Definição clara da aplicação.  
   - Uso de um conjunto de dados público contendo informações sobre nuvens de pontos.  
   - Análise exploratória para entender as características dos dados.  

2. **Treinamento do Modelo**  
   - Escolha do modelo **Random Forest** para classificação.  
   - Pré-processamento dos dados utilizando **MinMaxScaler**.  
   - Avaliação do desempenho com métricas como **Matriz de Confusão** e **Relatório de Classificação**.  

3. **Implementação do Endpoint**  
   - Desenvolvimento de uma API utilizando **Flask** ou **FastAPI**.  
   - O endpoint recebe os dados de entrada e retorna a classificação dos pontos.  

4. **Desenvolvimento do Frontend**  
   - Interface interativa para facilitar a visualização e interação com os resultados.  
   - Utilizando **Gradio**.  

5. **Documentação e Apresentação**  
   - Relatório detalhado sobre todas as etapas do projeto.  
   - Apresentação demonstrando os resultados e desafios enfrentados.  

📄 **[Documentação do projeto](https://docs.google.com/document/d/1GHl7TQAHpqj_4DYWphi3rCgld5yS4gNHdrcstA9DGBc/edit?usp=sharing)**  

📂 **[Drive com os dados das nuvens de pontos](https://drive.google.com/drive/folders/1s18AL6UCysrMWVYkRq8X7-eGyGP62YUe)**  


## Tecnologias Utilizadas
- **Linguagem**: Python  
- **Bibliotecas**:  
  - `pandas`, `numpy` → Manipulação de dados  
  - `matplotlib` → Visualização  
  - `sklearn` → Treinamento e avaliação do modelo  
  - `pickle` → Serialização do modelo  
  - `logging` → Registro de logs  

## Como Executar o Projeto
### Requisitos:
- Python 3.8+
- Bibliotecas mencionadas acima (podem ser instaladas com `pip install -r requirements.txt`)

### Execução:
1. Clone este repositório:  
```bash
   git clone https://github.com/gustavo-ghcs/AI_project.git
   cd AI_project/
   ```
2. Instale as dependências:
```bash
   pip install -r requirements.txt
   ```
3. Baixe as nuvens de treino e teste a partir do [drive de nuvens de pontos](https://drive.google.com/drive/folders/1s18AL6UCysrMWVYkRq8X7-eGyGP62YUe)e coloque-as na pasta `data/`:
- **Treino:** `3DML_urban_point_cloud.xyz`
- **Teste:** `nuvem_pontos_*.xyz.xyz`
     
4. Execute o script principal para treinar o modelo:
```bash
   python training.py
   ```
5. Os resultados da classificação serão armazenados no arquivo `data/classified_cloud_point.xyz` e o modelo em `models/model_trained.pkl`.

## Resultados Esperados

Após o treinamento, o modelo será capaz de classificar corretamente pontos em uma nuvem 3D nas categorias:

- Terreno (ground)
- Vegetação (vegetation)
- Edifícios (buildings)

A avaliação do modelo é feita com métricas padrão de aprendizado de máquina, como **precision**, **recall**, **F1-score** e **matriz de confusão**.

## Contato
Para dúvidas ou sugestões, entre em contato:

📧 Email: `ghcs@cin.ufpe.br`, `gmdn@cin.ufpe.br`, `lcmc@cin.ufpe.br`

🔗 GitHub: `gustavo-ghcs`, `geovannaadomingos`, `lucasccampos`
