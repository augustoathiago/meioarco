import numpy as np
import streamlit as st
import plotly.graph_objects as go

# =========================
# Constantes e Física
# =========================
EPS0 = 8.8e-12  # C²/(N·m²)

def clean_small(val, tol=1e-14):
    """
    Zera valores muito pequenos para evitar ruído numérico.
    """
    arr = np.asarray(val, dtype=float)
    arr = np.where(np.abs(arr) < tol, 0.0, arr)
    if arr.shape == ():
        return float(arr)
    return arr

def trig_deg_exact(theta_deg):
    """
    Retorna cos(theta) e sin(theta) usando theta em graus,
    com correção exata para ângulos especiais:
    0°, 90°, 180°, 270°, 360°.

    Isso evita erros como:
    sin(180°) ~ 1,22e-16 em vez de 0.
    """
    th = np.asarray(theta_deg, dtype=float)
    thn = np.mod(th, 360.0)

    c = np.cos(np.deg2rad(thn))
    s = np.sin(np.deg2rad(thn))

    tol = 1e-12

    # cos
    c = np.where(np.isclose(thn, 0.0, atol=tol), 1.0, c)
    c = np.where(np.isclose(thn, 90.0, atol=tol), 0.0, c)
    c = np.where(np.isclose(thn, 180.0, atol=tol), -1.0, c)
    c = np.where(np.isclose(thn, 270.0, atol=tol), 0.0, c)

    # sin
    s = np.where(np.isclose(thn, 0.0, atol=tol), 0.0, s)
    s = np.where(np.isclose(thn, 90.0, atol=tol), 1.0, s)
    s = np.where(np.isclose(thn, 180.0, atol=tol), 0.0, s)
    s = np.where(np.isclose(thn, 270.0, atol=tol), -1.0, s)

    c = clean_small(c)
    s = clean_small(s)

    if np.asarray(c).shape == ():
        return float(c), float(s)
    return c, s

def arc_length(a: float, theta_rad: float):
    """Comprimento do arco: L = a * theta (theta em rad)."""
    return a * theta_rad

def total_charge(lmbda: float, a: float, theta_rad: float):
    """Carga total do arco: Q = lambda * L."""
    return lmbda * arc_length(a, theta_rad)

def field_components_lambda(lmbda, a, theta_deg):
    """
    Campo elétrico no centro para um arco que:
    - começa no topo do círculo (0°)
    - cresce no sentido anti-horário até theta

    Fórmulas:
        Ex = (lambda / (4*pi*eps0*a)) * (1 - cos(theta))
        Ey = -(lambda / (4*pi*eps0*a)) * sin(theta)

    theta em graus, com tratamento exato de ângulos especiais.
    """
    lmbda_arr = np.asarray(lmbda, dtype=float)
    a_arr = np.asarray(a, dtype=float)
    theta_arr = np.asarray(theta_deg, dtype=float)

    shape = np.broadcast(lmbda_arr, a_arr, theta_arr).shape
    pref = np.zeros(shape, dtype=float)

    np.divide(
        lmbda_arr,
        4.0 * np.pi * EPS0 * a_arr,
        out=pref,
        where=(a_arr != 0)
    )

    cos_t, sin_t = trig_deg_exact(theta_arr)

    Ex = pref * (1.0 - cos_t)
    Ey = -pref * sin_t

    Ex = clean_small(Ex)
    Ey = clean_small(Ey)

    if np.asarray(Ex).shape == ():
        return float(Ex), float(Ey)
    return Ex, Ey

def field_magnitude(Ex, Ey):
    Emod = np.sqrt(np.asarray(Ex, dtype=float)**2 + np.asarray(Ey, dtype=float)**2)
    Emod = clean_small(Emod)
    if np.asarray(Emod).shape == ():
        return float(Emod)
    return Emod

def field_angle_deg(Ex: float, Ey: float):
    """
    Ângulo do vetor E em relação ao eixo +x,
    medido no sentido anti-horário, de 0° a 360°.
    """
    if np.isclose(Ex, 0.0, atol=1e-14) and np.isclose(Ey, 0.0, atol=1e-14):
        return None
    ang = np.degrees(np.arctan2(Ey, Ex))
    ang = (ang + 360.0) % 360.0
    return float(ang)

# =========================
# Formatação: 10^n (sem e±)
# =========================
def sci_parts(val: float, sig: int = 3):
    if val == 0:
        return 0.0, 0
    exp = int(np.floor(np.log10(abs(val))))
    mant = val / (10 ** exp)
    mant = float(f"{mant:.{sig-1}f}")
    if abs(mant) >= 10:
        mant /= 10
        exp += 1
    return mant, exp

def fmt_latex_10(val: float, unit: str = "", sig: int = 3):
    if val == 0:
        s = "0"
        return f"{s}\\,\\text{{{unit}}}" if unit else s
    mant, exp = sci_parts(val, sig=sig)
    mant_str = f"{mant:.{sig-1}f}".replace(".", ",")
    s = f"{mant_str}\\times 10^{{{exp}}}"
    return f"{s}\\,\\text{{{unit}}}" if unit else s

def fmt_html_10(val: float, unit: str = "", sig: int = 3):
    if val == 0:
        return f"0 {unit}".strip()
    mant, exp = sci_parts(val, sig=sig)
    mant_str = f"{mant:.{sig-1}f}".replace(".", ",")
    return f"{mant_str}×10<sup>{exp}</sup> {unit}".strip()

def fmt_dec_pt(val: float, nd: int = 3):
    return f"{val:.{nd}f}".replace(".", ",")

def sentido_x(Ex: float):
    if Ex > 0:
        return "para a direita", "→"
    if Ex < 0:
        return "para a esquerda", "←"
    return "nulo", "•"

def sentido_y(Ey: float):
    if Ey > 0:
        return "para cima", "↑"
    if Ey < 0:
        return "para baixo", "↓"
    return "nulo", "•"

# =========================
# Configuração da página
# =========================
st.set_page_config(
    page_title="Simulador Campo Elétrico de um Arco de Circunferência – Física II",
    layout="wide"
)

st.markdown(
    """
    <style>
    @media (max-width: 768px){
      div[data-testid="stPlotlyChart"] iframe,
      div[data-testid="stPlotlyChart"] > div {
        height: 500px !important;
        min-height: 500px !important;
      }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Cabeçalho
# =========================
c1, c2 = st.columns([1, 4], vertical_alignment="center")
with c1:
    try:
        st.image("logo_maua.png", use_container_width=True)
    except Exception:
        st.warning("Adicione **logo_maua.png** na pasta do app para exibir o logo.")
with c2:
    st.markdown(
        """
        # Simulador Campo Elétrico no Centro de um Arco de Circunferência – Física II
        **Estude o campo elétrico gerado por um arco carregado no ponto central P.**
        """
    )

st.divider()

# =========================
# Parâmetros
# =========================
st.subheader("Parâmetros")

A_MIN, A_MAX = 0.05, 1.00      # m
L_U_MIN, L_U_MAX = -20.0, 20.0 # µC/m
L_U_STEP = 0.1
THETA_MIN, THETA_MAX = 0, 360  # graus

colp1, colp2, colp3 = st.columns(3)

with colp1:
    lmbda_u = st.slider(
        "Densidade linear λ (µC/m)",
        min_value=float(L_U_MIN),
        max_value=float(L_U_MAX),
        value=2.0,
        step=float(L_U_STEP)
    )
    lmbda = lmbda_u * 1e-6  # C/m

with colp2:
    a = st.slider(
        "Raio a (m)",
        min_value=float(A_MIN),
        max_value=float(A_MAX),
        value=0.25,
        step=0.01
    )

with colp3:
    theta_deg = st.slider(
        "Ângulo do arco θ (graus)",
        min_value=int(THETA_MIN),
        max_value=int(THETA_MAX),
        value=180,
        step=1
    )

theta_rad = np.deg2rad(theta_deg)

# =========================
# Cálculos principais
# =========================
L = arc_length(a, theta_rad)
Q = total_charge(lmbda, a, theta_rad)

Ex, Ey = field_components_lambda(lmbda, a, theta_deg)
Emod = field_magnitude(Ex, Ey)
angE = field_angle_deg(Ex, Ey)

sx_text, sx_arrow = sentido_x(Ex)
sy_text, sy_arrow = sentido_y(Ey)

if Emod == 0:
    sentido_resultante = "nulo"
else:
    sentido_resultante = f"ângulo de {fmt_dec_pt(angE, 2)}° em relação ao eixo +x"

st.divider()

# =========================
# Escala global da imagem
# =========================
@st.cache_data(show_spinner=False)
def compute_global_emax_for_scene():
    lam_abs = max(abs(L_U_MIN), abs(L_U_MAX)) * 1e-6
    return float(1.15 * lam_abs / (2.0 * np.pi * EPS0 * A_MIN))

E_MAX_SCENE = compute_global_emax_for_scene()

# =========================
# Imagem
# =========================
st.subheader("Imagem")
st.caption("📱 No celular: arraste a figura para os lados (pan) para ver tudo sem perder detalhes.")

BASE = max(A_MAX, 1.8)
X_LEFT, X_RIGHT = -1.80 * BASE, 2.00 * BASE
Y_BOTTOM, Y_TOP = -1.50 * BASE, 1.50 * BASE

def add_vertical_dimension(fig, xdim, y0, y1, text):
    """Desenha uma cota vertical simples com traços horizontais nas extremidades."""
    tick = 0.06 * BASE
    fig.add_trace(go.Scatter(
        x=[xdim, xdim], y=[y0, y1],
        mode="lines",
        line=dict(color="black", width=2),
        hoverinfo="skip",
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[xdim - tick, xdim + tick], y=[y0, y0],
        mode="lines",
        line=dict(color="black", width=2),
        hoverinfo="skip",
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[xdim - tick, xdim + tick], y=[y1, y1],
        mode="lines",
        line=dict(color="black", width=2),
        hoverinfo="skip",
        showlegend=False
    ))
    fig.add_annotation(
        x=xdim - 0.12 * BASE, y=(y0 + y1) / 2,
        text=text,
        textangle=-90,
        showarrow=False,
        font=dict(size=12, color="black"),
        bgcolor="rgba(255,255,255,0.85)"
    )

def make_scene_figure(a, lmbda, Q, theta_deg, Ex, Ey, Emod):
    theta_rad_local = np.deg2rad(theta_deg)

    if Q > 0:
        arc_color = "red"
    elif Q < 0:
        arc_color = "blue"
    else:
        arc_color = "black"

    fig = go.Figure()

    # Eixos
    fig.add_trace(go.Scatter(
        x=[X_LEFT, X_RIGHT], y=[0, 0],
        mode="lines",
        line=dict(color="gray", dash="dash", width=2),
        hoverinfo="skip",
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[Y_BOTTOM, Y_TOP],
        mode="lines",
        line=dict(color="gray", dash="dash", width=2),
        hoverinfo="skip",
        showlegend=False
    ))

    # Círculo de referência
    tt = np.linspace(0, 2*np.pi, 700)
    x_full = a * np.cos(tt)
    y_full = a * np.sin(tt)
    fig.add_trace(go.Scatter(
        x=x_full, y=y_full,
        mode="lines",
        line=dict(color="rgba(120,120,120,0.18)", width=2),
        hoverinfo="skip",
        showlegend=False
    ))

    # Arco selecionado
    if theta_deg > 0:
        th = np.linspace(0, theta_rad_local, max(60, int(2.2 * theta_deg) + 2))
        arc_x = -a * np.sin(th)
        arc_y =  a * np.cos(th)

        fig.add_trace(go.Scatter(
            x=arc_x, y=arc_y,
            mode="lines",
            line=dict(color=arc_color, width=5),
            hoverinfo="skip",
            showlegend=False
        ))

        # ponto inicial e final do arco
        fig.add_trace(go.Scatter(
            x=[0], y=[a],
            mode="markers",
            marker=dict(size=8, color=arc_color),
            hoverinfo="skip",
            showlegend=False
        ))

        x_end_arc = -a * np.sin(theta_rad_local)
        y_end_arc =  a * np.cos(theta_rad_local)
        fig.add_trace(go.Scatter(
            x=[x_end_arc], y=[y_end_arc],
            mode="markers",
            marker=dict(size=8, color=arc_color),
            hoverinfo="skip",
            showlegend=False
        ))
    else:
        fig.add_trace(go.Scatter(
            x=[0], y=[a],
            mode="markers",
            marker=dict(size=8, color="black"),
            hoverinfo="skip",
            showlegend=False
        ))

    # ponto P
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers+text",
        marker=dict(size=12, color="black"),
        text=["P"],
        textposition="top center",
        hoverinfo="skip",
        showlegend=False
    ))

    # Vetores Ex, Ey, E
    max_arrow_len = 0.34 * (X_RIGHT - X_LEFT)
    min_arrow_len = 0.08 * (X_RIGHT - X_LEFT)

    frac = 0.0 if E_MAX_SCENE == 0 else min(1.0, abs(Emod) / E_MAX_SCENE)
    arrow_len = 0.0 if Emod == 0 else (min_arrow_len + (max_arrow_len - min_arrow_len) * np.sqrt(frac))

    if Emod > 0:
        scale = arrow_len / Emod
        dx = Ex * scale
        dy = Ey * scale

        # Ex
        fig.add_annotation(
            x=dx, y=0,
            ax=0, ay=0,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.0,
            arrowwidth=3,
            arrowcolor="#ff7f0e"
        )

        # Ey
        fig.add_annotation(
            x=dx, y=dy,
            ax=dx, ay=0,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.0,
            arrowwidth=3,
            arrowcolor="#9467bd"
        )

        # E resultante
        fig.add_annotation(
            x=dx, y=dy,
            ax=0, ay=0,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.1,
            arrowwidth=4,
            arrowcolor="green"
        )

        # rótulos
        fig.add_annotation(
            x=dx/2 if abs(dx) > 1e-12 else 0.12*BASE, y=0.08*BASE,
            text="<b>E<sub>x</sub></b>",
            showarrow=False,
            font=dict(size=13, color="#ff7f0e"),
            bgcolor="rgba(255,255,255,0.85)"
        )
        fig.add_annotation(
            x=dx + 0.09*BASE*np.sign(dx if abs(dx) > 1e-12 else 1),
            y=dy/2 if abs(dy) > 1e-12 else -0.12*BASE,
            text="<b>E<sub>y</sub></b>",
            showarrow=False,
            font=dict(size=13, color="#9467bd"),
            bgcolor="rgba(255,255,255,0.85)"
        )
        fig.add_annotation(
            x=0.60*dx, y=0.60*dy + 0.06*BASE,
            text="<b>E</b>",
            showarrow=False,
            font=dict(size=13, color="green"),
            bgcolor="rgba(255,255,255,0.85)"
        )

    # Cota vertical do raio à esquerda
    xdim = -a - 0.18 * BASE
    add_vertical_dimension(
        fig,
        xdim=xdim,
        y0=0,
        y1=a,
        text=f"a = {fmt_dec_pt(a, 3)} m"
    )

    # Caixa λ e θ
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=(
            f"λ = {fmt_html_10(lmbda, 'C/m', sig=3)}<br>"
            f"θ = {theta_deg}°"
        ),
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=12, color="black")
    )

    # Caixa do campo
    fig.add_annotation(
        x=0.98, y=0.98, xref="paper", yref="paper",
        text=(
            f"<b>E<sub>x</sub></b> = {fmt_html_10(Ex, 'N/C', sig=3)}<br>"
            f"<b>E<sub>y</sub></b> = {fmt_html_10(Ey, 'N/C', sig=3)}<br>"
            f"<b>|E|</b> = {fmt_html_10(Emod, 'N/C', sig=3)}"
        ),
        showarrow=False,
        xanchor="right",
        yanchor="top",
        align="left",
        bgcolor="rgba(255,255,255,0.94)",
        bordercolor="rgba(0,0,0,0.55)",
        borderwidth=1,
        font=dict(size=12, color="black")
    )

    # Títulos dos eixos
    fig.add_annotation(
        x=X_RIGHT - 0.04*(X_RIGHT-X_LEFT), y=0.06*BASE,
        text="<b>x</b>",
        showarrow=False,
        font=dict(size=14, color="black")
    )
    fig.add_annotation(
        x=0.05*BASE, y=Y_TOP - 0.05*(Y_TOP-Y_BOTTOM),
        text="<b>y</b>",
        showarrow=False,
        font=dict(size=14, color="black")
    )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        dragmode="pan"
    )
    fig.update_xaxes(range=[X_LEFT, X_RIGHT], visible=False, fixedrange=False)
    fig.update_yaxes(
        range=[Y_BOTTOM, Y_TOP],
        visible=False,
        scaleanchor="x",
        scaleratio=1,
        fixedrange=True
    )

    return fig

scene = make_scene_figure(a, lmbda, Q, theta_deg, Ex, Ey, Emod)
st.plotly_chart(
    scene,
    use_container_width=True,
    config={"scrollZoom": False, "displayModeBar": False, "responsive": True}
)

st.divider()

# =========================
# Equações
# =========================
st.subheader("Equações")

st.markdown("**Comprimento do arco**")
st.latex(r"L = a\,\theta \qquad (\theta \text{ em rad})")

st.markdown("**Carga total**")
st.latex(r"Q = \lambda L = \lambda a \theta")

st.markdown("**Componentes do campo elétrico no centro**")
st.markdown(
    "Adotando o ângulo do arco medido **a partir do topo** e aumentando no **sentido anti-horário**:"
)

st.latex(r"E_x = \frac{\lambda}{4\pi\varepsilon_0 a}\int_0^{\theta}\sin(\theta')\,d\theta'")
st.latex(r"E_y = -\,\frac{\lambda}{4\pi\varepsilon_0 a}\int_0^{\theta}\cos(\theta')\,d\theta'")

st.markdown("**Módulo do campo elétrico**")
st.latex(r"|E| = \sqrt{E_x^2 + E_y^2}")

st.markdown("**Permissividade do vácuo**")
st.latex(r"\varepsilon_0 = 8,8\times10^{-12}\ \text{C}^2/\text{N·m}^2")

st.divider()

# =========================
# Cálculos
# =========================
st.subheader("Cálculos")

st.latex(
    rf"\theta = {theta_deg}^\circ = {fmt_dec_pt(theta_rad, 4)}\ \text{{rad}}"
)

st.latex(
    rf"L = a\theta = ({fmt_dec_pt(a,3)})({fmt_dec_pt(theta_rad,4)}) = {fmt_latex_10(L,'m',sig=4)}"
)

st.latex(
    rf"Q = \lambda L = \left({fmt_latex_10(lmbda,'C/m',sig=3)}\right)\left({fmt_latex_10(L,'m',sig=4)}\right)"
    rf" = {fmt_latex_10(Q,'C',sig=4)}"
)

# Ex
st.latex(
    r"E_x = \frac{\lambda}{4\pi\varepsilon_0 a}\int_0^{\theta}\sin(\theta')\,d\theta'"
)

st.latex(
    rf"E_x = \frac{{{fmt_latex_10(lmbda,'C/m',sig=3)}}}{{4\pi(8,8\times10^{{-12}})({fmt_dec_pt(a,3)})}}"
    rf"\left[-\cos(\theta')\right]_{{0^\circ}}^{{{theta_deg}^\circ}}"
)

st.latex(
    rf"E_x = \frac{{{fmt_latex_10(lmbda,'C/m',sig=3)}}}{{4\pi(8,8\times10^{{-12}})({fmt_dec_pt(a,3)})}}"
    rf"\left(\cos 0^\circ - \cos {theta_deg}^\circ\right)"
)

st.latex(
    rf"E_x = {fmt_latex_10(Ex,'N/C',sig=4)}\quad {sx_arrow}"
)
st.markdown(f"**Sentido de \(E_x\):** {sx_text}")

# Ey
st.latex(
    r"E_y = -\,\frac{\lambda}{4\pi\varepsilon_0 a}\int_0^{\theta}\cos(\theta')\,d\theta'"
)

st.latex(
    rf"E_y = -\,\frac{{{fmt_latex_10(lmbda,'C/m',sig=3)}}}{{4\pi(8,8\times10^{{-12}})({fmt_dec_pt(a,3)})}}"
    rf"\left[\sin(\theta')\right]_{{0^\circ}}^{{{theta_deg}^\circ}}"
)

st.latex(
    rf"E_y = -\,\frac{{{fmt_latex_10(lmbda,'C/m',sig=3)}}}{{4\pi(8,8\times10^{{-12}})({fmt_dec_pt(a,3)})}}"
    rf"\left(\sin {theta_deg}^\circ - \sin 0^\circ\right)"
)

st.latex(
    rf"E_y = {fmt_latex_10(Ey,'N/C',sig=4)}\quad {sy_arrow}"
)
st.markdown(f"**Sentido de \(E_y\):** {sy_text}")

# Módulo
st.latex(r"|E| = \sqrt{E_x^2 + E_y^2}")
st.latex(
    rf"|E| = \sqrt{{\left({fmt_latex_10(Ex,'N/C',sig=4)}\right)^2 + \left({fmt_latex_10(Ey,'N/C',sig=4)}\right)^2}}"
)
st.latex(
    rf"|E| = {fmt_latex_10(Emod,'N/C',sig=4)}"
)

# Ângulo do vetor
if angE is None:
    st.markdown("**Ângulo do campo elétrico:** campo nulo (ângulo indefinido).")
else:
    st.latex(
        rf"\alpha_E = \operatorname{{atan2}}(E_y,E_x) = {fmt_dec_pt(angE,2)}^\circ"
    )
    st.markdown(f"**Direção do vetor \(\vec E\):** {sentido_resultante}")

st.divider()

# =========================
# Gráficos
# =========================
st.subheader("Gráficos")

def curve_Emod_vs_a(lmbda, theta_deg):
    aas = np.linspace(A_MIN, A_MAX, 450)
    Ex_arr, Ey_arr = field_components_lambda(lmbda, aas, theta_deg)
    E_arr = field_magnitude(Ex_arr, Ey_arr)
    return aas, E_arr

def curve_Emod_vs_Q(a, theta_deg):
    theta_rad_local = np.deg2rad(theta_deg)
    L_local = arc_length(a, theta_rad_local)

    if np.isclose(L_local, 0.0, atol=1e-15):
        Qs = np.array([-1e-12, 0.0, 1e-12])
        Es = np.array([0.0, 0.0, 0.0])
        return Qs, Es, -1e-12, 1e-12

    Qmin = (L_U_MIN * 1e-6) * L_local
    Qmax = (L_U_MAX * 1e-6) * L_local

    if np.isclose(Qmin, Qmax, atol=1e-20):
        Qmin -= 1e-12
        Qmax += 1e-12

    Qs = np.linspace(Qmin, Qmax, 450)
    lambdas = Qs / L_local
    Ex_arr, Ey_arr = field_components_lambda(lambdas, a, theta_deg)
    E_arr = field_magnitude(Ex_arr, Ey_arr)
    return Qs, E_arr, Qmin, Qmax

def style_axes_black(fig):
    fig.update_xaxes(
        title_font=dict(color="black"),
        tickfont=dict(color="black"),
        showline=True, linecolor="black",
        ticks="outside", tickcolor="black",
        exponentformat="power",
        automargin=True,
        fixedrange=True
    )
    fig.update_yaxes(
        title_font=dict(color="black"),
        tickfont=dict(color="black"),
        showline=True, linecolor="black",
        ticks="outside", tickcolor="black",
        exponentformat="power",
        automargin=True,
        fixedrange=True
    )
    return fig

aas, Ea = curve_Emod_vs_a(lmbda, theta_deg)
Qs, EQ, Q_MIN_AXIS, Q_MAX_AXIS = curve_Emod_vs_Q(a, theta_deg)

max_val = float(np.max(np.concatenate([
    np.asarray(Ea, dtype=float),
    np.asarray(EQ, dtype=float),
    np.array([Emod], dtype=float)
])))
if max_val == 0:
    max_val = 1.0
YMAX = 1.08 * max_val

PLOT_CFG_STATIC = {
    "staticPlot": True,
    "displayModeBar": False,
    "scrollZoom": False,
    "responsive": True
}

gx1, gx2 = st.columns(2)

with gx1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=aas, y=Ea,
        mode="lines",
        line=dict(color="#2ca02c", width=3)
    ))
    fig1.add_trace(go.Scatter(
        x=[a], y=[Emod],
        mode="markers",
        marker=dict(color="red", size=10)
    ))
    fig1.update_layout(
        title="Módulo do campo elétrico em função do raio a",
        title_font=dict(color="black"),
        height=430,
        margin=dict(l=60, r=18, t=65, b=95),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False
    )
    fig1.update_xaxes(title="a (m)", range=[A_MIN, A_MAX], zeroline=True)
    fig1.update_yaxes(title="|E| (N/C)", range=[0, YMAX], zeroline=True)
    style_axes_black(fig1)
    st.plotly_chart(fig1, use_container_width=True, config=PLOT_CFG_STATIC)

with gx2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=Qs, y=EQ,
        mode="lines",
        line=dict(color="#9467bd", width=3)
    ))
    fig2.add_trace(go.Scatter(
        x=[Q], y=[Emod],
        mode="markers",
        marker=dict(color="red", size=10)
    ))
    fig2.update_layout(
        title="Módulo do campo elétrico em função da carga total Q",
        title_font=dict(color="black"),
        height=430,
        margin=dict(l=60, r=18, t=65, b=95),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False
    )
    fig2.update_xaxes(title="Q (C)", range=[Q_MIN_AXIS, Q_MAX_AXIS], zeroline=True)
    fig2.update_yaxes(title="|E| (N/C)", range=[0, YMAX], zeroline=True)
    style_axes_black(fig2)
    st.plotly_chart(fig2, use_container_width=True, config=PLOT_CFG_STATIC)

st.caption("🔴 O ponto vermelho indica a situação atual.")

# =========================
# Observação final
# =========================
with st.expander("Convenção angular usada no app"):
    st.markdown(
        """
        Neste simulador, o arco é definido por um único ângulo \\(\\theta\\):
        - **0°**: topo do círculo
        - o arco cresce no **sentido anti-horário**
        - o campo elétrico \\(\\vec E\\) é mostrado com:
          - componente **x**: \\(E_x\\)
          - componente **y\\)**: \\(E_y\\)
          - módulo: \\(|E|\\)
          - ângulo do vetor em relação ao eixo **+x**
        """
    )
