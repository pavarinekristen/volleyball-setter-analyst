# Plano de Adaptação: Basquete → Levantamento de Vôlei

## 0. Validação da análise prévia (com base na leitura integral do código)

Antes do plano, ponto a ponto da hipótese inicial, confirmando ou corrigindo com `arquivo:linha`.

**Confirmado, quase 100% reaproveitável sem alteração:**
- `src/pose.py` — genérico: usa só `KPT_NAMES` (COCO-17), nenhuma lógica de basquete. `build_request`, `unwrap`, `request_poses`, `cache_path`, `dedupe` são 100% independentes de domínio.
- `src/video.py` — probe/convert/segment/encode, tudo genérico de ffmpeg/OpenCV. Zero acoplamento a basquete.
- `src/env.py` — carregamento de API key, genérico.
- `src/skeleton.py` — esqueleto COCO-17 e cores por membro; genérico (serve para qualquer esporte com pose humana).

**Parcialmente correto — refinamento necessário:**
- `src/hoop.py`: "mecânica de cache reaproveitável" é verdade, mas mais preciso: `request_segment`, `read_frame`, `cache_path` são wrappers genéricos do SAM "segment" — 100% reaproveitáveis tal como estão (só troca o prompt). Mas a classe `Hoop` e a função `locate()` (hoop.py:88-117) — que acham o aro pelas "linhas mais largas da máscara" — são semântica de aro e **não** se generalizam; precisam de uma classe/função equivalente para o que for a nova âncora (rede, por exemplo).
- `src/ball.py`: é **mais reaproveitável do que a hipótese inicial supôs** — `BallTrack`, `typical_diameter`, `select()`, `smoothed()`, `request_track`, `detections`, `cache_path` não têm *nenhuma* referência a basquete; é um "SAM-track + filtro por tamanho/redondeza" 100% genérico (ball.py:1-181). A única coisa realmente basquete-específica que toca esse módulo está em `main.py`: a função `track_ball()` (main.py:159-194), que decide **quando e por quanto tempo** chamar o SAM com base nos spans de `candidates()` e na regra "ainda está perto do aro, estende a janela" (`shots_mod.ball_unresolved`, shots.py:285-291, usa `hoop.net_bottom`). Isso precisa ser reescrito — a localização exata do código é essa.

**Confirmado — reescrita necessária:**
- `shots.py: candidates()` (shots.py:75-106): gatilho "ambos os punhos acima da cabeça, armado quando caem abaixo do ombro/`armed_below`", com `_median` suavizando 5 frames e `min_gap_seconds` evitando duplo-conta do follow-through. O teste assume uma pose sustentada (o arremesso de basquete "para" no alto por alguns frames); o toque de levantamento é momentâneo (<0.1–0.2s) e pode nem elevar os punhos acima da linha da cabeça de forma limpa (toque de dedos na testa/acima da testa, não necessariamente "acima do topo da cabeça" como o teste mede via `HEAD` = nariz/olhos/orelhas, shots.py:13). Também: recepção de saque com manchete fica com as mãos abaixo da cintura (não dispara), mas um levantamento de ataque "overhead" com as duas mãos, ou um bloqueio, ou um saque por cima, podem disparar esse teste quase do mesmo jeito.
- `shots.py: judge()` (shots.py:208-282): as 5 regras (vem de cima, cruza dentro do aro, está na distância do aro, perde velocidade na rede, não volta a subir) dependem inteiramente de `Hoop.cx/cy/width/net_bottom`. Não há conceito equivalente em um levantamento — não existe "entra ou não entra". Precisa ser substituído por métricas de qualidade.
- `court.py`: a função `project()` (court.py:27-39) reaproveita a matemática pinhole (focal `f`, profundidade `z_hoop` via largura real/pixel do aro, profundidade `z_player` via altura ombro-tornozelo/pixel), só troca `hoop.width`/`rim_diameter_m` pela âncora nova. Mas atenção: **`project()` só devolve posição horizontal no chão (x, y), nunca altura vertical (z) de um ponto acima do chão.** "Altura do contato em metros" **não tem nenhum precedente no código de basquete**: nada ali calcula altura absoluta em metros (só graus de ângulo, ou "altura de soltura em comprimentos de tronco" relativos, em `form.py:105-109`). Isso é trabalho novo, não um "trocar a âncora" — fórmula na seção 5.

**Confirmado:**
- `render.py` trilha da bola (`BALL_TRAIL_SECONDS`, ClipOverlay.draw, render.py:163-200) e `plots.png`/`trajectory.py` — a arquitetura visual (caminho desenhado, cores por resultado, dispersão no "shot chart") é o esqueleto certo, mas precisa de ajustes de janela (seção 7).
- `form.py` (ângulo do cotovelo 2D, só válido de perfil) — reaproveitável como está para "braço esticado vs. dobrado no contato", trocando apenas o instante medido (contato, não "release") e possivelmente os dois braços (levantamento é geralmente com os dois).
- Detecção de salto via deslocamento de tornozelo/quadril, adaptando `shots.py: stance()` (shots.py:294-311) — correto, e resolve também parte do problema de calibração de altura.

---

## 1. Mapa módulo-a-módulo

| Módulo | Decisão | Justificativa |
|---|---|---|
| `src/pose.py` | **Reaproveita 100%** | Zero acoplamento a basquete; já é genérico ViTPose/Gateway. |
| `src/video.py` | **Reaproveita 100%** | Probe/convert/segment/encode são genéricos. |
| `src/env.py` | **Reaproveita 100%** | API key, genérico. |
| `src/skeleton.py` | **Reaproveita 100%** | COCO-17, genérico. |
| `src/ball.py` | **Reaproveita 100%** (código); troca só `BALL_PROMPT`/`BALL_DIAMETER_M` em config | `BallTrack`, `select()`, `smoothed()` são agnósticos ao objeto rastreado. |
| `src/hoop.py` | **Reescrita parcial**: mantém `request_segment`/`read_frame`/`cache_path` (genéricos de SAM-segment); descarta `Hoop` + `locate()` | A semântica "rim = linhas mais largas da máscara" não existe para rede/altura de jogador. |
| `src/court.py` | **Reescrita parcial**: mantém a matemática pinhole de `project()`; `CourtView` (desenho) é 100% descartado (quadra de basquete) | A trigonometria é genérica; a geometria/desenho da quadra é específica do esporte. Precisa de função nova para altura vertical (z). |
| `src/shots.py` | **Reescrita quase total**: mantém utilidades genéricas (`keypoints()`, `elbow_angle()`, `shoulder_ratio()`, `_median()`, o padrão de `motion_start()`/`find_release()`); descarta `candidates()`, `judge()`, `Shot`, `ball_unresolved()` | A detecção de evento e o julgamento são o núcleo específico de basquete; as funções de geometria 2D de pose são genéricas. |
| `src/render.py` | **Modificação pontual pesada**: mantém `ClipOverlay`/`draw_ball`/trilha/`plot_elbow` como esqueleto; descarta `draw_rim`, hot-streak (`_ball_fire`, `_rim_heat`, `MagmaText`), `plot_summary`'s painel de aro | Trilha e gráfico de "arm angle through time" são reaproveitáveis; tudo relacionado a aro/streak não existe no vôlei. |
| `src/panel.py` | **Modificação pontual**: mantém infraestrutura de texto/fonte/crédito/`_spread`; descarta FG/FG%, hot streak, `CourtView` de basquete | Infra de desenho de HUD é genérica; conteúdo é 100% basquete. |
| `src/export.py` | **Reescrita pontual**: mesmo padrão CSV/JSON, campos novos | Estrutura reaproveitável, campos trocam. |
| `src/text.py` | **Reaproveita 100%** | Fonte TrueType, genérico. |
| `form.py` | **Reescrita pontual**: reaproveita `joint_angle()`, teste de perfil (`shoulder_ratio`), teste de permutação (`permutation_p`) | Métrica muda de "cotovelo no set/release" para "extensão no contato"; estatística é genérica. |
| `trajectory.py` | **Reescrita pontual**: a técnica de ajuste parabólico (`fit()`, `flight()`) é genérica para qualquer objeto em queda livre | Precisa trocar a referência (não há "rim" para parar o ajuste; troca para "momento do contato" como início). |
| `rim.py` | **Descartado** | Inteiramente sobre geometria do aro/rede de basquete; nenhum conceito equivalente direto. |
| `history.py`, `highlights.py`, `social.py`, `compilation.py` | **Reescrita pontual de média/baixa prioridade** | Reaproveitam o padrão de "carrega cache, agrega sessões, plota", mas todo conteúdo (zonas de quadra, FG%, swish) precisa ser trocado por variância/consistência. Não bloqueante para o MVP. |

---

## 2. Novo detector de evento de contato

### 2.0 Pré-requisito arquitetural (decisão que muda tudo)

No basquete, o pipeline só manda a bola pro SAM **dentro das janelas** que `candidates()` (pose) já decidiu (main.py:159-194, `track_ball()`). É uma economia de custo: SAM só é chamado perto de cada arremesso. "Detectar contato pela trajetória da bola" tem um problema de ovo-e-galinha: **não há trajetória da bola para analisar fora das janelas, porque a bola só é rastreada dentro delas.** Decidir isso antes de desenhar o detector. Três arquiteturas:

- **Opção A — sessão de treino isolado (toque, levantamento, repete):** vídeo é um drill focado (análogo ao "shooting session" do basquete) — a sessão inteira pode ser rastreada continuamente pelo SAM, sem gating por pose. Mais simples, mas custo de gateway escala com a duração total do clipe, não com o número de eventos.
- **Opção B — jogo/treino misto:** precisa de um filtro grosseiro primeiro (pose) para não pagar SAM no vídeo inteiro.
- **Opção C — híbrida (recomendada):** um primeiro passe de SAM *barato* (stride maior, ex. cada 2–3 frames) rastreia a bola o tempo todo, só para achar "onde a bola sobe e desce perto de uma pessoa" grosseiramente; depois um segundo passe, *fino* (stride 1), re-rastreia só nas janelas curtas candidatas para achar o instante exato do contato. Generaliza o padrão que já existe em `main.py: track_ball()` (janela + extensões quando "ainda não resolvido"), só que a "janela inicial" vem de um passe grosseiro do próprio SAM, não da pose.

**Decisão do usuário necessária antes de codar** — qual é o tipo de vídeo real? Recomendação default: Opção C, mas confirmar com o usuário primeiro.

### 2.1 Desenho do detector (duas camadas)

**Camada 1 — candidatos grosseiros (pose, frouxo, alta recall/baixa precisão):**
Reaproveitar o esqueleto de `candidates()` (shots.py:75-106) com teste mais permissivo: punhos acima de uma linha mais baixa que "topo da cabeça" (ex. acima dos ombros ou do nariz, configurável), sem exigir sustentação de 5 frames (reduzir a janela de suavização de `_median`, evento é curtíssimo), sem a regra de "armado" baseada em descer abaixo do ombro (toques se repetem rápido). Vai gerar **muito mais falsos positivos** que o basquete (recepção com braços erguidos, bloqueio, ataque, saque por cima) — esperado, o papel dessa camada é só reduzir o vídeo a trechos candidatos.

**Camada 2 — validação e localização fina (trajetória da bola + geometria):**
Dentro de cada janela candidata, com a bola já rastreada por SAM (stride 1):
1. Calcular `vy[f] = y[f+1] - y[f]` (velocidade vertical da bola, pixels/frame; y crescente para baixo).
2. Contato válido = **cruzamento de sinal de `vy`**: bola descendo (`vy > 0`) passa a subir (`vy < 0`), ou pico local abrupto de desaceleração. Interpolação linear entre as duas amostras que cercam a troca de sinal — mesmo padrão já usado em `shots.py: judge()` para o cruzamento sub-frame da linha do aro (shots.py:231-234).
3. **Proximidade geométrica:** no frame do cruzamento, distância da bola ao(s) punho(s)/dedos menor que ~1–1.5 diâmetros de bola — reaproveitando `shots_mod.motion_start()` (shots.py:167-187, já implementa "bola a X diâmetros do punho"). Sem essa checagem, qualquer pico de `vy` (bola batendo no chão, saindo de quadro) seria falso positivo.
4. **Filtro de duração do contato:** troca de sinal deve ocorrer num intervalo curto (≤ 2–3 frames); posse prolongada é sinal de erro de rastreio ou toque ilegal — descartar/marcar suspeito.
5. **Filtro de pose no instante do contato:** validar padrão de braço consistente com toque de dedos (cotovelos flexionados, mãos próximas acima da cabeça, tronco vertical). Ajuda a distinguir de manchete e parcialmente de ataque, mas **não resolve de forma confiável a distinção com bloqueio**, toque de uma mão, ou ataque de segunda bola. Risco técnico real — ver seção 9.

### 2.2 Riscos de confusão previsíveis

- **Manchete/recepção baixa:** baixo risco — mãos abaixo do peito/cabeça, Camada 1 já exclui a maior parte.
- **Bloqueio:** alto risco — duas mãos, acima da cabeça, perto da rede, braços estendidos. Mitigação parcial: no bloqueio a bola normalmente não "toca e sai" com inversão clara de sinal perto do punho (é desviada ou nem toca), então "vy inverte E bola para quase exatamente nos punhos" tende a não disparar tão fácil. Se vídeo for de jogo real, considerar filtro de posição no quadro (levantamentos mais distantes da rede que bloqueios).
- **Ataque com toque de dedos / "set" de ataque:** alto risco de confusão genuína — mecanicamente quase idêntico a um levantamento normal. Só distinguível por contexto de jogo (1ª, 2ª ou 3ª bola), fora do escopo de uma v1; limitação conhecida.
- **Saque por cima:** risco baixo-médio — uma mão só, bola vem de toss do próprio jogador; validável com regra extra de geometria de quadra (saque ocorre atrás da linha de fundo), mas difícil de generalizar.
- **Outros toques de dedos (ressalto, defesa, cortina):** aceitável tratar qualquer "toque de dedos acima da cabeça com trajetória pós-contato ascendente e pra frente" como candidato, documentando a limitação.

**Recomendação prática:** para v1, assumir Opção A/drill controlado (reduz drasticamente os riscos); desenhar Camada 1+2 já pensando em v2 com flag `--review` sinalizando eventos de confiança baixa para revisão manual.

---

## 3. Calibração de profundidade/escala sem aro

### 3.1 Altura do jogador — recomendado como primário
`shots.py: stance()` (shots.py:294-311) já resolve o problema de flexão de joelho: mede o tamanho ombro-tornozelo pelo **percentil 90 ao longo de ~2s antes do evento**, pegando o jogador "em pé e esticado" mesmo com agachamento no meio. Para o levantamento, mesma técnica funciona, com a ressalva de que a profundidade (z) precisa ser calculada num instante em que o jogador não mudou de posição horizontal — razoável assumir que o salto é majoritariamente vertical, permitindo reusar o mesmo z para calibração e medição no mesmo evento.
- Vantagem: código quase pronto, reaproveita `stance()` direto.
- Desvantagem: exige janela "em pé" limpa; exige `PLAYER_HEIGHT_M` informado (já é assim no basquete via `--player-height`).

### 3.2 Rede — bom complemento, não substituto único
Vantagem sobre o aro: atravessa o quadro inteiro, calibra a cena toda. Riscos concretos:
- **Visibilidade:** em ângulos focados no levantador, a rede pode estar parcial ou totalmente fora de quadro.
- **Segmentação SAM da rede:** malha com muitos buracos é alvo de segmentação mais difícil que um aro sólido — risco de máscara ruidosa/fragmentada. Prompt provável: "volleyball net" ou "net cable/tape".
- **Combado da rede:** fita superior tem ~2cm de combado no centro; medir ali introduz viés. Mitigação: amostrar perto das laterais/antenas.
- Só dá profundidade (z) e escala horizontal no plano da própria rede — mesma extrapolação de "escala varia com 1/z" do resto do sistema, não é matematicamente mais simples que o aro, só potencialmente mais robusta espacialmente.

**Recomendação:** usar como calibração secundária/validação cruzada quando visível, não como dependência única.

### 3.3 Diâmetro da bola — ótimo para o instante do contato, não para escala geral
A bola, no instante do contato (ou frames de voo livre antes/depois), dá a profundidade exatamente no ponto que mais importa (`z = f × diâmetro_real / diâmetro_px`), reaproveitando `ball_mod.typical_diameter()` (ball.py:98-103), já usado localmente por janela em `select(local_size=True)` (ball.py:106-158).
- Vantagem: mede profundidade no próprio ponto de interesse.
- Desvantagem: frame exato do contato tem maior risco de blur/oclusão pelas mãos — usar os 2-3 frames imediatamente antes do contato (bola ainda em voo livre).

**Recomendação final de calibração:** altura do jogador como base, diâmetro da bola pré-contato como validação cruzada, rede quando visível como terceira fonte opcional. Reportar as três no JSON de saída e expor divergência como "confiança da calibração" — não existe hoje no basquete (só uma fonte, o aro), é código novo.

---

## 4. Novo pipeline passo-a-passo (equivalente a `run_clip()`)

1. **Fonte e conversão** (`video.probe_source`, `video.convert`) — sem mudança.
2. **Pose: ViTPose no clipe inteiro** — sem mudança na chamada; muda o que é feito com o resultado.
3. **Candidatos grosseiros de evento** (novo `sets_mod.candidates()`, Camada 1) — substitui `shots_mod.candidates()`. Saída: janelas largas (ex. ±1s) de possíveis contatos.
4. **Calibração base** (novo `calib_mod.find_anchor()`): tenta achar a rede via SAM (1 chamada, análoga a `find_hoop`), e/ou prepara lógica de altura-do-jogador. Diferença chave: a âncora pode falhar (rede fora de quadro) sem quebrar o pipeline — precisa de fallback gracioso.
5. **Rastreio de bola** (Opção C): passe grosseiro contínuo + passe fino dentro das janelas candidatas. Reaproveita `ball_mod.request_track`/`detections`/`select` quase sem mudança de código, só orquestração em `main.py`.
6. **Detecção fina do contato + validação** (Camada 2) — novo módulo, substitui `find_release`/`judge`.
7. **Métricas por evento** (novo `Set` dataclass — seção 5) — substitui `analyze()` + `Shot`.
8. **Agregação de sessão** (novo, não existe no basquete): médias, desvios-padrão, scatter — módulo novo `src/consistency.py`.
9. **Render** (vídeo anotado, gráficos) — `render.py` modificado (seção 7).
10. **Export CSV/JSON/run.json** — `export.py` modificado.

**Chamadas de API que mudam de forma:**
- SAM "segment" numa rede em vez de aro: mesmo `method: "segment"`, prompt novo, mas `locate()` precisa de lógica geométrica diferente (achar a linha superior da fita, não a "linha mais larga").
- SAM "track" da bola: com Opção C, duas chamadas por trecho (passe grosseiro stride alto + passe fino stride 1), em vez de uma janela por evento como no basquete — implicação direta de custo.
- ViTPose: sem mudança de chamada.

---

## 5. Novas estruturas de dados: `Set` e métricas de qualidade

```python
@dataclass
class Set:
    number: int
    contact_frame: float          # instante do contato, sub-frame (interpolado)
    window: tuple[int, int]       # janela de rastreio bruta usada
    approach_start: int | None    # início do movimento de aproximação/salto

    # pose / forma
    side_on: bool
    arm_extended_deg: float       # ângulo do(s) cotovelo(s) no contato (180 = reto)
    jumped: bool
    jump_height_m: float | None   # deslocamento vertical do quadril/tornozelo vs. baseline em pé

    # altura
    contact_height_m: float       # altura absoluta do ponto de contato (mãos) sobre o piso
    ideal_height_m: float | None  # alcance máximo do próprio jogador (braço esticado, em pé ou saltando)
    height_deficit_m: float       # ideal_height_m - contact_height_m (negativo = acima do esperado)

    # calibração usada (rastreabilidade)
    calibration_source: str       # "player_height" | "net" | "ball_diameter" | "disagreement"
    calibration_confidence: float # 0-1, baseado na concordância entre fontes

    # posição no "court" (planta baixa)
    feet_px: tuple[float, float] | None
    court_xy: tuple[float, float] | None   # metros, relativo a uma origem fixa (ex. centro da rede)

    # trajetória
    path: list[tuple[float, float, float]]   # (frame, x_px, y_px), pré e pós-contato
    peak_height_post_m: float | None         # altura máxima da bola após o contato
    exit_angle_deg: float | None             # ângulo de saída da bola (lançamento)
    exit_direction_deg: float | None         # direção horizontal de saída (p/ qual zona de ataque)

    confidence: float   # confiança geral da detecção do evento (Camada 2)
```

**Fórmula de `contact_height_m`** (trabalho novo, sem análogo no basquete):
```
z = profundidade do jogador no evento (metros; de 3.1, 3.2 ou 3.3)
f = (largura_do_quadro / 2) / tan(HFOV / 2)                         # já existe em court.project()
escala_vertical_m_por_px = z / f                                    # metros por pixel nessa profundidade
contact_height_m = (y_px_pe - y_px_contato) * escala_vertical_m_por_px
```
onde `y_px_pe` é a posição vertical do tornozelo/piso no mesmo instante (ou interpolada de um instante próximo em que o jogador está parado, se o salto tirar o pé do chão).

**`ideal_height_m`**: medir, numa janela separada (um instante de alongamento espontâneo, ou o alcance máximo observado em qualquer frame da sessão com o braço esticado verticalmente acima da cabeça, em pé ou no pico do salto), usando a mesma fórmula. Reaproveita o padrão de `form.py: measure()` que já busca extremos de ângulo numa janela (ex. `elbow_release` = `nanmax`, form.py:97) — aqui generalizado para achar o "pico de alcance" em toda a sessão, não só por evento.

**`SessionConsistency`** (agregado novo, `src/consistency.py`):
```python
@dataclass
class SessionConsistency:
    n_sets: int
    contact_point_mean_xy: tuple[float, float]
    contact_point_std_xy: tuple[float, float]       # desvio padrão espacial do ponto de contato
    contact_height_mean_m: float
    contact_height_std_m: float
    height_deficit_mean_m: float
    jump_rate: float                                 # fração de levantamentos em que saltou
    trajectory_peak_mean_m: float
    trajectory_peak_std_m: float
    exit_direction_std_deg: float                     # dispersão da direção de saída
    exit_direction_mean_deg: float
```
100% código novo — o basquete nunca precisa de desvio-padrão porque sua métrica é binária (acerto/erro) agregada como percentual, não dispersão espacial/temporal contínua.

---

## 6. Mapeamento de `config.py`

**Morrem (sem equivalente):**
`HOOP_PROMPT`, `HOOP_FRAME_SECONDS`, `RIM_DIAMETER_M`, `RIM_INNER_FRACTION`, `NET_MIN_SECONDS`, `NET_SIDE_TOLERANCE`, `REBOUND_MARGIN_RIMS`, `RIM_GAP_SECONDS`, `FRONT_OF_RIM_SIZE`, `THREE_POINT_RADIUS_M`, `FREE_THROW_DISTANCE_M`, `HOT_STREAK`, `BALL_FIRE_SECONDS`, `BALL_FIRE_FADE_SECONDS`, `RIM_COOL_SECONDS`, `RIM_WARM`, `STREAK_FADE`, `SHOT_ARMED_BELOW` (substituído por lógica de camadas), `SHOOTING_HAND` (vôlei é geralmente bimanual).

**Mudam de valor/escopo (mesmo nome ou renomeado, mesma função):**
- `BALL_PROMPT` → `"volleyball"` / `"volleyball ball"`.
- `BALL_DIAMETER_M` (0.239) → ~0.21 m (bola FIVB oficial).
- `BALL_BEFORE_SECONDS`/`BALL_AFTER_SECONDS` → drasticamente menores (toque é instantâneo; ~0.3–0.5s em vez de 0.6/2.4s).
- `BALL_STRIDE` → possivelmente 2 valores agora (grosseiro/fino) em vez de um só.
- `SIDE_ON_MAX_SHOULDER_RATIO`, `ELBOW_MIN_PROJECTED`, `SET_WINDOW_SECONDS` → mantidos conceitualmente; `SET_WINDOW_SECONDS` precisa de nome novo pra não colidir com "o set" do vôlei (sugestão: `FORM_WINDOW_SECONDS`).
- `CAMERA_HFOV_DEGREES`, `PLAYER_HEIGHT_M`, `COURT_YAW_DEGREES` → sem mudança (genéricos de câmera pinhole).
- `BALL_TRAIL_SECONDS`, `BALL_FILL_SECONDS` → mantidos, valores talvez ajustados para cobrir pré-contato.
- `OUTPUT_CRF`, `CONVERT_CRF`, `INFERENCE_HEIGHT`, `EXPORT_HEIGHT`, `TONEMAP*` → sem mudança.

**Novas:**
- `NET_PROMPT` (ex. `"volleyball net"`), `NET_HEIGHT_M` (2.43 masc / 2.24 fem / 2.35 misto — dict configurável), `NET_WIDTH_M` (9.0 ou 9.5 m com antenas).
- `SET_MIN_GAP_SECONDS` (equivalente a `SHOT_MIN_GAP_SECONDS`).
- `CONTACT_HEAD_MARGIN` (limiar da Camada 1, frouxo).
- `CONTACT_PROXIMITY_BALLS` (equivalente a `HELD_REACH`, raio de proximidade bola-punho no contato).
- `CONTACT_MAX_HOLD_FRAMES` (filtro de posse prolongada = toque ilegal/erro de rastreio).
- `JUMP_THRESHOLD_M` (deslocamento mínimo para contar como salto).
- `IDEAL_REACH_WINDOW_SECONDS` (janela de busca do alcance máximo "ideal").
- `BALL_COARSE_STRIDE` (stride do passe grosseiro de rastreio contínuo, Opção C).

---

## 7. Mudanças em renderização e saída

**Vídeo anotado:**
- Mantém: esqueleto desenhado, trilha da bola com cometa (`draw_ball`), estendida para cobrir também o trecho pré-contato.
- Remove: `draw_rim` (acende verde/vermelho), toda lógica de "hot streak" (`_ball_fire`, `_rim_heat`, `MagmaText`, `Panel.streak`).
- Novo: marcador visual no instante do contato (círculo/retângulo na posição real), colorido por `height_deficit_m` (verde = próximo do ideal, vermelho = muito abaixo), opcionalmente uma "zona ideal" fixa desenhada como referência horizontal.

**Painel inferior (substitui o "shot chart"):**
- Reaproveita `Panel`/`CourtView` como padrão de código (desenho de planta + dispersão), vista top-down nova sem linhas de quadra de basquete — grid de referência com posição média e elipse de desvio-padrão (1-2σ) do ponto de contato.
- Métricas no canto: "altura média / desvio" e "nº de levantamentos" em vez de "FG / FG%".

**Gráficos equivalentes a `plots.png`:**
- Painel esquerdo: trajetória pré/pós-contato de todos os levantamentos sobrepostas, alinhadas pelo instante do contato (reaproveitando a técnica de normalização temporal de `elbow.png`), coloridas por proximidade à altura ideal.
- Painel direito: scatter do ponto de contato em planta baixa, com elipse de desvio-padrão, reaproveitando `_spread()` (panel.py:121-145).

**Gráfico equivalente a `elbow.png`:**
- Mesma estrutura (`plot_elbow`, render.py:344-452): curva do ângulo do cotovelo alinhada pelo contato ao invés do "release", tira de frames (`ELBOW_STRIP`) mostrando braço e ângulo — reaproveitável quase 1:1.

**Novo gráfico (sem análogo direto em basquete):** painel de consistência — altura do contato por levantamento em sequência cronológica (técnica de `rim.py` para "long/short" shot-by-shot, rim.py:419-462), linha de média móvel e banda de desvio-padrão, destacando deriva ao longo da sessão.

**CSV/JSON (`export.py`):**
- `*_sets.csv`: uma linha por levantamento com `contact_height_m`, `ideal_height_m`, `height_deficit_m`, `jumped`, `jump_height_m`, `court_x_m`, `court_y_m`, `arm_extended_deg`, `peak_height_post_m`, `exit_angle_deg`, `exit_direction_deg`, `calibration_source`, `confidence`.
- `*_frames.csv`: mantém o padrão, campos ajustados.
- `session_consistency.json`: **novo arquivo**, sem análogo no basquete, com o agregado `SessionConsistency`.
- `run.json`: mesmo papel, com bloco `calibration` (fonte usada, confiança, valores das 3 fontes) em vez de `hoop`.

---

## 8. Arquivos novos propostos e arquivos existentes a modificar

**Novos:**
- `src/net.py` — segmentação da rede via SAM e extração da linha superior (substitui papel estrutural de `hoop.py`).
- `src/sets.py` (ou renomear `shots.py`) — detecção de evento (Camadas 1+2), estrutura `Set`, métricas de forma/altura/salto.
- `src/calibration.py` — reconciliação das 3 fontes de profundidade/escala, com índice de confiança.
- `src/consistency.py` — agregação de sessão (`SessionConsistency`).
- `src/volley_court.py` (ou renomear `court.py` mantendo `project()`) — desenho de planta baixa do vôlei (zonas, linha de ataque, rede) e nova função de altura vertical (`height_m()`).

**Modificados (mudança pontual):**
- `main.py` — `run_clip()` reescrito seguindo o pipeline da seção 4; `track_ball()` reescrito para a arquitetura de 2 passes.
- `config.py` — conforme seção 6.
- `src/ball.py` — nenhuma mudança de lógica; só prompt/config.
- `src/render.py` — remove hot-streak e `draw_rim`; adapta `ClipOverlay` para o novo marcador de contato e trilha estendida; `plot_summary`/`plot_elbow` adaptados conforme seção 7.
- `src/panel.py` — remove conteúdo de FG/streak; mantém infraestrutura de texto/crédito/`_spread`.
- `src/export.py` — novos campos conforme seção 7.
- `form.py` — métrica de "extensão no contato" em vez de "set/release"; mantém `permutation_p`, `joint_angle`.
- `trajectory.py` — ajuste do parabolóide usando "contato" como referência em vez de "rim"; mantém `fit()`/`steady()`.
- `history.py`, `highlights.py`, `social.py`, `compilation.py` — baixa prioridade; reescrever depois do MVP.

**Descartados por completo:**
- `rim.py` — nenhum conceito equivalente direto.

---

## 9. Riscos, ambiguidades em aberto e esforço estimado

### Ambiguidades que precisam de decisão do usuário antes de implementar
1. **Tipo de vídeo-fonte**: drill isolado (toque-levantamento repetido) vs. jogo/treino misto com outras ações no quadro. Decide a arquitetura inteira da Camada 1 e o custo de rastreio de bola. **Decisão mais importante, deveria ser respondida primeiro.**
2. **A rede estará visível no enquadramento típico?** Decide se a calibração por rede é viável como fonte principal ou só eventual bônus.
3. **Levantamento é sempre bimanual no uso real, ou cobrir toques de emergência de uma mão?** Afeta a métrica de "braço esticado".
4. **"Altura ideal" é o alcance máximo histórico da sessão, um valor antropométrico calculado, ou informado manualmente?** Muda se `ideal_height_m` é medido ou configurado.
5. **Gênero/categoria da rede (2.43/2.24/2.35m)?** Necessário se a rede for usada como âncora.
6. **Quantas câmeras/ângulos serão usados?** O código assume uma câmera fixa, como no basquete; múltiplos ângulos é outro projeto.

### Riscos técnicos
- **Custo de gateway**: rastreio contínuo (mesmo em stride grosseiro) de sessões longas pode superar em muito o custo do basquete (que só paga janelas curtas por arremesso). Testar custo real antes de escalar.
- **Confusão de ação** (seção 2.2): bloqueio e ataque de segunda bola são geometricamente parecidos com o levantamento; sem contexto de jogo, falsos positivos esperados.
- **Segmentação da rede via SAM**: malha com buracos é alvo difícil; pode precisar de pós-processamento (dilatação morfológica, como em `rim.py: board_outline()`, rim.py:257-287) ou prompt mais específico.
- **Calibração de altura sem aro**: nenhuma das 3 fontes é tão robusta quanto o aro; esperar mais ruído/incerteza, expor isso via campo `confidence` em vez de escondê-lo.
- **Falta de "ground truth" de acerto/erro**: o aparato estatístico do basquete (make/miss, FG%, permutation test comparando dois grupos) não se aplica direto — a análise de vôlei é sobre dispersão contínua. `form.py: permutation_p` continua útil, mas para comparar sessões/jogadores entre si, não makes vs. misses.

### Estimativa qualitativa de esforço por bloco
| Bloco | Esforço |
|---|---|
| Infra (pose/video/env/skeleton) | Nenhum — já pronto |
| `ball.py` | Nenhum — já pronto, só config |
| Detector de evento (Camadas 1+2) | **Alto** — lógica nova, mais a decisão arquitetural da seção 2.0 |
| Calibração (rede/altura/bola) | **Alto** — três fontes, reconciliação, sem precedente direto de "altura absoluta" no código-base |
| `Set` + métricas de forma/salto | **Médio** — reutiliza bastante (`stance`, `elbow_angle`, `motion_start`), mas fórmulas novas de altura |
| `SessionConsistency` / agregação | **Médio-baixo** — estatística simples, mas conceito inteiramente novo |
| Render de vídeo (trilha, marcador de contato) | **Médio** — reaproveita bastante estrutura, remove bastante coisa |
| Gráficos (plots/elbow/consistência) | **Médio** — reaproveita bastante técnica, conteúdo novo |
| CSV/JSON | **Baixo** — mesmo padrão, campos novos |
| Scripts auxiliares (`history`/`highlights`/`social`/`compilation`) | **Baixa prioridade, esforço médio quando feito** — não bloqueante para MVP |
| `rim.py` / `trajectory.py` | Descartar / Reescrita pontual |

---

## Arquivos críticos do projeto original (referência)
- `main.py`
- `config.py`
- `src/shots.py`
- `src/ball.py`
- `src/hoop.py`
- `src/court.py`
- `src/render.py`
