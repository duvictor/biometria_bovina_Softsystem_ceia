## Biometria Bovina - SoftSystem - CEIA - UFG
# Identificação Bovina através de Reconhecimento Facial 📷🐄

Este repositório é dedicado ao desenvolvimento e manutenção de um sistema de **identificação bovina baseado em reconhecimento facial**. A ideia principal é utilizar técnicas avançadas de **Deep Learning** e **Visão Computacional** para identificar bovinos a partir de imagens de seus focinhos, que possuem características únicas como impressões digitais.

---

## 📌 Objetivo
1. Criar um pipeline robusto para realizar identificação de bovinos com precisão e eficiência.
2. Testar e comparar diferentes abordagens e modelos arquiteturais para extração de características biométricas.
3. Documentar o desempenho, resultados e experimentos realizados.
4. Armazenar tarefas, melhorias e evoluções relacionadas ao projeto.

---

## 🚀 Funcionalidades

### 1. **Cadastro de Bovinos**
- Geração de um **embedding biométrico** para cada bovino, utilizando modelos de Deep Learning como **ResNet**, **InceptionV3**, **MobileNetV2** e outros.
- Armazenamento do embedding, juntamente com as informações da vaca, em um banco de dados específico.

### 2. **Reconhecimento em Tempo Real**
- Captura de imagem ao vivo e **computação do embedding** usando a mesma arquitetura.
- Comparação do embedding gerado com os dados armazenados no banco para identificar o animal.
- Algoritmos de similaridade, como **Cosine Similarity** e **distância euclidiana**, são utilizados como métricas.

### 3. **Treinamento e Ajuste Fino**
- Implementação de técnicas como **Triplet Loss** e **Contrastive Loss** para otimizar a geração e a separação dos embeddings.
- Suporte a diferentes estratégias de treinamento e adaptação a variações de imagem (iluminação, escala, etc.).

### 4. **Visualização e Análise**
- Geração e visualização dos embeddings como mapas de calor e gráficos 2D/3D para análise.
- Aplicação de técnicas como PCA e t-SNE para validação e detecção de possíveis duplicações ou outliers nos embeddings.

---

## 📂 Estrutura do Projeto

- **Códigos Fonte:** Scripts de treinamento, teste e geração de embeddings.
- **Modelos Pré-treinados:** Implementação de arquiteturas como MobileNetV2 e ResNet50.
- **Documentação:** Explicação das funções, métodos e pipeline geral do projeto.
- **Resultados e Métricas:** Testes realizados e benchmarking de diferentes abordagens.
- **Tarefas:** Registro de atividades a serem desenvolvidas e melhorias planejadas.

---

## 🛠️ Tecnologias Utilizadas
- **Linguagem de Programação:** Python 3.11
- **Frameworks de Machine Learning:** 
  - TensorFlow e Keras
  - Scikit-learn (para análise e métricas)
- **Manipulação de Dados:** NumPy, Pandas
- **Visualização:** Matplotlib, Seaborn
- **Infraestrutura:**
  - GPUs para aceleração do treinamento
  - Suporte para implementação em sistemas de borda (aplicações IoT rurais)
  
---

## 📝 Pipeline Básico

### 1. Captura e Pré-processamento de Dados
- Captura de imagens dos focinhos dos bovinos.
- Redimensionamento e normalização para o modelo de entrada.

### 2. Extração de Características (Backbone)
Os seguintes modelos são recomendados para extração eficiente de características biométricas:
- **ResNet-50/101:** Ideal para padrões profundos e invariantes.
- **InceptionV3:** Captura detalhes em imagens com variações de textura e escala.
- **MobileNetV2:** Arquitetura leve e rápida, ideal para dispositivos embarcados.
- **EfficientNet:** Eficiência balanceada entre custo computacional e precisão.
  
### 3. Treinamento do Modelo
- Funções de perda:
  - **Triplet Loss:** Focado em maximizar a similaridade entre embeddings do mesmo indivíduo e minimizar entre indivíduos diferentes.
  - **Contrastive Loss:** Mede a similaridade entre pares de imagens com base em um rótulo binário.

### 4. Identificação
- Com base no embedding gerado, a distância é calculada em relação ao banco de dados:
  - Animais mais próximos ao embedding atual serão considerados como possíveis correspondências.
- Limiar de aceitação configurável, preferencialmente ajustado durante a validação.

---

## 🎯 Chamadas Principais no Pipeline

### Construção do Modelo
```python
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2

def build_backbone(model_name='mobilenetv2', input_shape=(224, 224, 3)):
    if model_name == 'mobilenetv2':
        base = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    # Adaptação adicional para normalização e pooling
    x = GlobalAveragePooling2D()(base.output)
    x = tf.math.l2_normalize(x, axis=1)
    return Model(inputs=base.input, outputs=x)
```

### Função de Perda (Exemplo: Triplet Loss)
```python
def triplet_loss(margin=0.5):
    def loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:, 0, :], y_pred[:, 1, :], y_pred[:, 2, :]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
    return loss
```

### Treinamento
```python
# Exemplo usando Triplet Model
triplet_model.compile(optimizer='adam', loss=triplet_loss())
triplet_model.fit([anchors, positives, negatives], np.zeros(len(anchors)), epochs=10)
```

---

## 🧬 Casos de Uso Planejados

1. **Cadastro Inicial:**
   - Capturar imagens ao cadastrar a vaca. Armazenar o embedding e informações no banco.

2. **Reconhecimento Automático:**
   - Utilizar câmeras em fazendas para identificação em tempo real.

3. **Manutenção Contínua do Banco:**
   - Atualizar embeddings com novas imagens ao longo do tempo.

---

## 📊 Testes e Resultados
- Benchmarking dos modelos (`MobileNetV2`, `ResNet`, etc.) para medir:
  - Tempo de processamento
  - Taxa de acurácia e erro
- Testes realizados para diferentes condições, como variações na iluminação, ângulo e sujeiras leves no focinho.

---

## 🔮 Futuras Melhorias
1. **Aprimoramento de Precisão:**
   - Testar técnicas híbridas, como ensembles de modelos.
2. **Integração IoT:**
   - Adaptação para processamentos de borda com dispositivos integrados.
3. **Interface do Usuário:**
   - Implementação de dashboards para visualização e monitoramento em tempo real.

---

## 🤝 Contribuições
Contribuições, sugestões e melhorias são sempre bem-vindas! Sinta-se à vontade para abrir um *issue* ou enviar um *pull request*.

---

## 🐾 Agradecimentos
Este projeto está em desenvolvimento para auxiliar no manejo e controle zootécnico de animais, promovendo tecnologias modernas ao agronegócio. 🌱
CEIA - UFG
