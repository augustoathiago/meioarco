import numpy as np
import streamlit as st
import plotly.graph_objects as go

# =========================
# Constantes e Física
# =========================
EPS0 = 8.8e-12  # C²/(N·m²)
K = 1.0 / (4.0 * np.pi * EPS0)

def arc_length(a: float) -> float:
    """Comprimento de um meio arco."""
    return np.pi * a

def total_charge(lmbda: float, a: float) -> float:
    """Carga total do meio arco."""
    return lmbda * arc_length(a)

def numerical_trapezoid(y, x, axis=-1):
    """
    Integração robusta por trapézios.
    Usa np.trapezoid se existir; caso contrário, usa fallback manual.
    """
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)

    # fallback manual
    y = np.asarray(y)
    x = np.asarray(x)

    dx = np.diff(x)
    sl1 = [slice(None)] * y.ndim
    sl2 = [slice(None)] * y.ndim
    sl1[axis] = slice(1, None)
    sl2[axis] = slice(None, -1)

    y1 = y[tuple(sl1)]
    y2 = y[tuple(sl2)]

    shape_dx = [1] * y.ndim
    shape_dx[axis] = dx.shape[0]
    dx_reshaped = dx.reshape(shape_dx)

    return np.sum((y1 + y2) * 0.5 * dx_reshaped, axis=axis)

def field_on_axis_semicircle(x, a: float, lmbda: float, n_theta: int = 3000):
    """
    Campo elétrico Ex gerado por um meio arco com boca virada para a direita,
    isto é, o arco está na metade esquerda do círculo.

    Parametrização:
        r'(theta) = (-a cos(theta), a sin(theta)), theta em [-pi/2, pi/2]

    Ponto:
        P = (x, 0), com x >= 0

    Pela simetria:
        Ey = 0
        Ex é o campo resultante ao longo do eixo horizontal

    Fórmula integral:
        Ex = k * lambda * a * ∫ [ (x + a cos(theta)) /
                                 (x² + a² + 2ax cos(theta))^(3/2) ] dtheta
             com theta em [-pi/2, pi/2]
    """
    x_arr = np.asarray(x, dtype=float)
    scalar_input = (x_arr.ndim == 0)
    if scalar_input:
        x_arr = x_arr[None]

    theta = np.linspace(-np.pi / 2, np.pi / 2, n_theta)
    c = np.cos(theta)

    # Distância ao quadrado do elemento de carga até P
    # x' = -a cos(theta), y' = a sin(theta)
    # R² = (x - x')² + (0 - y')² = (x + a cos(theta))² + (a sin(theta))²
    #    = x² + a² + 2ax cos(theta)
    R2 = x_arr[:, None]**2 + a*a + 2.0*a*x_arr[:, None]*c[None, :]
    numer = x_arr[:, None] + a*c[None, :]

    integrand = numer / np.power(R2, 1.5)
    ex = K * lmbda * a * numerical_trapezoid(integrand, theta, axis=1)

    return float(ex[0]) if scalar_input else ex

def field_center_formula(Q: float, a: float) -> float:
    """
    Campo no centro (x=0) para o meio arco com boca virada para a direita.
    Para Q > 0, Ex > 0.
    """
    if a == 0:
        return 0.0
    return Q / (2.0 * np.pi**2 * EPS0 * a*a)


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


# =========================
# Configuração da página
# =========================
st.set_page_config(
    page_title="Simulador Campo Elétrico do Meio Arco – Física II",
    layout="wide"
)

# CSS responsivo: gráficos mais altos no celular
st.markdown(
    """
    <style>
    @media (max-width: 768px){
      div[data-testid="stPlotlyChart"] iframe,
      div[data-testid="stPlotlyChart"] > div {
        height: 440px !important;
        min-height: 440px !important;
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
        # Simulador Campo Elétrico do Meio Arco – Física II
        **Estude o campo elétrico gerado por um meio arco carregado em um ponto P sobre o eixo horizontal.**
        """
    )

st.divider()

# =========================
# Parâmetros
# =========================
st.subheader("Parâmetros")

X_MIN, X_MAX = 0.00, 2.00     # m
A_MIN, A_MAX = 0.05, 1.00     # m

# λ em µC/m
L_U_MIN, L_U_MAX = -20.0, 20.0
L_U_STEP = 0.1

colp1, colp2, colp3 = st.columns(3)

with colp1:
    x = st.slider(
        "Distância x (m)",
        min_value=float(X_MIN),
        max_value=float(X_MAX),
        value=0.40,
        step=0.01
    )

with colp2:
    lmbda_u = st.slider(
        "Densidade linear λ (µC/m)",
        min_value=float(L_U_MIN),
        max_value=float(L_U_MAX),
        value=2.0,
        step=float(L_U_STEP)
    )
    lmbda = lmbda_u * 1e-6  # C/m

with colp3:
    a = st.slider(
        "Raio a (m)",
        min_value=float(A_MIN),
        max_value=float(A_MAX),
        value=0.25,
        step=0.01
    )

# Cálculos
L = arc_length(a)
Q = total_charge(lmbda, a)
Ex = field_on_axis_semicircle(x, a, lmbda)

# Sentido do campo
if Ex > 0:
    sentido_seta = "→"
    sentido_texto = "para a direita"
elif Ex < 0:
    sentido_seta = "←"
    sentido_texto = "para a esquerda"
else:
    sentido_seta = "•"
    sentido_texto = "nulo"

st.divider()

# =========================
# Emax global (escala do vetor na imagem)
# =========================
@st.cache_data(show_spinner=False)
def compute_global_emax_for_scene():
    xs = np.linspace(X_MIN, X_MAX, 180)
    aas = np.linspace(A_MIN, A_MAX, 90)
    lam_abs = max(abs(L_U_MIN), abs(L_U_MAX)) * 1e-6

    emax = 1.0
    for aa in aas:
        E = field_on_axis_semicircle(xs, aa, lam_abs, n_theta=1200)
        local_max = float(np.max(np.abs(E)))
        if local_max > emax:
            emax = local_max
    return 1.15 * emax

E_MAX_SCENE = compute_global_emax_for_scene()

# =========================
# Imagem
# =========================
st.subheader("Imagem")
st.caption("📱 No celular: arraste a figura para os lados (pan) para ver tudo sem perder detalhes.")

BASE = max(A_MAX, X_MAX)
X_LEFT, X_RIGHT = -1.30 * BASE, 2.10 * BASE
Y_LIM = 1.25 * A_MAX

def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))

def make_scene_figure(x, a, lmbda, Q, Ex):
    # cor do arco
    if Q > 0:
        arc_color = "red"
    elif Q < 0:
        arc_color = "blue"
    else:
        arc_color = "black"

    fig = go.Figure()

    # eixo horizontal tracejado
    fig.add_trace(go.Scatter(
        x=[X_LEFT, X_RIGHT], y=[0, 0],
        mode="lines",
        line=dict(color="gray", dash="dash", width=2),
        hoverinfo="skip",
        showlegend=False
    ))

    # meio arco com boca virada para a direita (arco na metade esquerda)
    t = np.linspace(-np.pi/2, np.pi/2, 500)
    arc_x = -a * np.cos(t)
    arc_y = a * np.sin(t)

    fig.add_trace(go.Scatter(
        x=arc_x, y=arc_y,
        mode="lines",
        line=dict(color=arc_color, width=5),
        hoverinfo="skip",
        showlegend=False
    ))

    # centro
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers",
        marker=dict(size=8, color="black"),
        hoverinfo="skip",
        showlegend=False
    ))

    # ponto P
    fig.add_trace(go.Scatter(
        x=[x], y=[0],
        mode="markers+text",
        marker=dict(size=12, color="black"),
        text=["P"],
        textposition="top center",
        hoverinfo="skip",
        showlegend=False
    ))

    # =========================
    # Vetor E
    # =========================
    max_arrow_len = 0.28 * (X_RIGHT - X_LEFT)
    min_arrow_len = 0.08 * (X_RIGHT - X_LEFT)
    frac = 0.0 if E_MAX_SCENE == 0 else min(1.0, abs(Ex) / E_MAX_SCENE)
    arrow_len = min_arrow_len + (max_arrow_len - min_arrow_len) * np.sqrt(frac)
    dx = arrow_len if Ex >= 0 else -arrow_len

    x_end = x + dx
    x_end = clamp(x_end, X_LEFT + 0.03*(X_RIGHT-X_LEFT), X_RIGHT - 0.03*(X_RIGHT-X_LEFT))

    fig.add_annotation(
        x=x_end, y=0,
        ax=x, ay=0,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.1,
        arrowwidth=4,
        arrowcolor="green"
    )

    # Caixa do campo
    box_x = clamp(x + 0.18*BASE, X_LEFT + 0.20*BASE, X_RIGHT - 0.55*BASE)
    box_y = 0.38 * Y_LIM

    fig.add_annotation(
        x=box_x, y=box_y,
        text=f"<b>E</b> = {fmt_html_10(Ex, 'N/C', sig=3)}",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        align="left",
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="rgba(0,0,0,0.55)",
        borderwidth=1,
        font=dict(size=13, color="green")
    )

    # =========================
    # Cota x
    # =========================
    y_dimx = -0.35 * Y_LIM
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[0, y_dimx],
        mode="lines",
        line=dict(color="black", width=1),
        hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[x, x], y=[0, y_dimx],
        mode="lines",
        line=dict(color="black", width=1),
        hoverinfo="skip", showlegend=False
    ))
    fig.add_annotation(
        x=x, y=y_dimx,
        ax=0, ay=y_dimx,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.0,
        arrowwidth=2,
        arrowcolor="black",
        text=""
    )
    fig.add_annotation(
        x=(x/2 if x > 0 else 0.08*BASE), y=y_dimx - 0.06*Y_LIM,
        text=f"x = {fmt_dec_pt(x, 3)} m",
        showarrow=False,
        font=dict(color="black", size=12)
    )

    # =========================
    # Cota a
    # =========================
    ang = 3*np.pi/4  # raio apontando para o meio do arco à esquerda
    xr = a * np.cos(ang)
    yr = a * np.sin(ang)

    fig.add_annotation(
        x=xr, y=yr,
        ax=0, ay=0,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=0,
        arrowsize=1.0,
        arrowwidth=2,
        arrowcolor="black"
    )
    fig.add_annotation(
        x=0.62 * xr - 0.02*BASE, y=0.62 * yr + 0.05*Y_LIM,
        text=f"a = {fmt_dec_pt(a, 3)} m",
        showarrow=False,
        font=dict(color="black", size=12)
    )

    # =========================
    # Caixa fixa com λ
    # =========================
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=f"λ = {fmt_html_10(lmbda, 'C/m', sig=3)}",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=12, color="black")
    )

    # Layout
    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        dragmode="pan"
    )
    fig.update_xaxes(range=[X_LEFT, X_RIGHT], visible=False, fixedrange=False)
    fig.update_yaxes(range=[-Y_LIM, Y_LIM], visible=False, scaleanchor="x", scaleratio=1, fixedrange=True)

    return fig

scene = make_scene_figure(x, a, lmbda, Q, Ex)
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

st.markdown("**Arco de circunferência**")
st.latex(r"L = \pi a")

st.markdown("**Carga total**")
st.latex(r"Q = \lambda\,L = \lambda(\pi a)")

st.markdown("**Campo elétrico no eixo horizontal**")
st.latex(
    r"E_x = \frac{1}{4\pi\varepsilon_0}\int_{-\pi/2}^{\pi/2}"
    r"\frac{\lambda a\,(x+a\cos\theta)}{\left(a^2+x^2+2ax\cos\theta\right)^{3/2}}\,d\theta"
)

st.markdown("**No centro do meio arco \((x=0)\)**")
st.latex(r"E_x(0) = \frac{Q}{2\pi^2\varepsilon_0 a^2}")

st.markdown("**Permissividade do vácuo**")
st.latex(r"\varepsilon_0 = 8,8\times10^{-12}\ \text{C}^2/\text{N·m}^2")

st.divider()

# =========================
# Cálculos
# =========================
st.subheader("Cálculos")

st.latex(rf"L = \pi a = \pi({fmt_dec_pt(a,3)}) = {fmt_latex_10(L,'m',sig=4)}")

st.latex(
    rf"Q = \lambda L = \left({fmt_latex_10(lmbda,'C/m',sig=3)}\right)\left({fmt_latex_10(L,'m',sig=4)}\right)"
    rf" = {fmt_latex_10(Q,'C',sig=4)}"
)

st.latex(rf"\varepsilon_0 = 8,8\times10^{{-12}}\ \text{{C}}^2/\text{{N·m}}^2")

st.markdown("**Campo elétrico calculado numericamente pela integral acima:**")
st.latex(rf"E_x = {fmt_latex_10(Ex,'N/C',sig=4)}\quad {sentido_seta}")

if abs(x) < 1e-12:
    Ex_center = field_center_formula(Q, a)
    st.markdown("**Como \(x=0\), também vale a expressão fechada do centro:**")
    st.latex(
        rf"E_x(0) = \frac{{Q}}{{2\pi^2\varepsilon_0 a^2}}"
        rf" = {fmt_latex_10(Ex_center,'N/C',sig=4)}\quad {sentido_seta}"
    )

st.divider()

# =========================
# Gráficos
# =========================
st.subheader("Gráficos")

def curve_E_vs_x(a, lmbda):
    xs = np.linspace(X_MIN, X_MAX, 280)
    E = field_on_axis_semicircle(xs, a, lmbda, n_theta=1400)
    return xs, E

def curve_E_vs_a(x, lmbda):
    aas = np.linspace(A_MIN, A_MAX, 240)
    E = np.array([field_on_axis_semicircle(x, aa, lmbda, n_theta=1400) for aa in aas])
    return aas, E

def curve_E_vs_Q(x, a):
    # Para a e x fixos, Q varia linearmente com λ: Q = λ π a
    Qmin = (L_U_MIN * 1e-6) * np.pi * a
    Qmax = (L_U_MAX * 1e-6) * np.pi * a
    Qs = np.linspace(Qmin, Qmax, 240)
    lambdas = Qs / (np.pi * a)
    E = np.array([field_on_axis_semicircle(x, a, lam, n_theta=1400) for lam in lambdas])
    return Qs, E, Qmin, Qmax

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

xs, Es = curve_E_vs_x(a, lmbda)
aas, Ea = curve_E_vs_a(x, lmbda)
Qs, EQ, Q_MIN_AXIS, Q_MAX_AXIS = curve_E_vs_Q(x, a)

max_abs = float(np.max(np.abs(np.concatenate([Es, Ea, EQ]))))
if max_abs == 0:
    max_abs = 1.0
YMAX = 1.08 * max_abs

PLOT_CFG_STATIC = {
    "staticPlot": True,
    "displayModeBar": False,
    "scrollZoom": False,
    "responsive": True
}

gx1, gx2, gx3 = st.columns(3)

with gx1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=xs, y=Es, mode="lines", line=dict(color="#1f77b4", width=3)))
    fig1.add_trace(go.Scatter(x=[x], y=[Ex], mode="markers", marker=dict(color="red", size=10)))
    fig1.update_layout(
        title="Campo elétrico em função da distância x",
        title_font=dict(color="black"),
        height=430,
        margin=dict(l=60, r=18, t=65, b=95),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False
    )
    fig1.update_xaxes(title="x (m)", range=[X_MIN, X_MAX], zeroline=True)
    fig1.update_yaxes(title="Eₓ (N/C)", range=[-YMAX, YMAX], zeroline=True)
    style_axes_black(fig1)
    st.plotly_chart(fig1, use_container_width=True, config=PLOT_CFG_STATIC)

with gx2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=aas, y=Ea, mode="lines", line=dict(color="#2ca02c", width=3)))
    fig2.add_trace(go.Scatter(x=[a], y=[Ex], mode="markers", marker=dict(color="red", size=10)))
    fig2.update_layout(
        title="Campo elétrico em função do raio a",
        title_font=dict(color="black"),
        height=430,
        margin=dict(l=60, r=18, t=65, b=95),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False
    )
    fig2.update_xaxes(title="a (m)", range=[A_MIN, A_MAX], zeroline=True)
    fig2.update_yaxes(title="Eₓ (N/C)", range=[-YMAX, YMAX], zeroline=True)
    style_axes_black(fig2)
    st.plotly_chart(fig2, use_container_width=True, config=PLOT_CFG_STATIC)

with gx3:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=Qs, y=EQ, mode="lines", line=dict(color="#9467bd", width=3)))
    fig3.add_trace(go.Scatter(x=[Q], y=[Ex], mode="markers", marker=dict(color="red", size=10)))
    fig3.update_layout(
        title="Campo elétrico em função da carga total Q",
        title_font=dict(color="black"),
        height=430,
        margin=dict(l=60, r=18, t=65, b=95),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False
    )
    fig3.update_xaxes(title="Q (C)", range=[Q_MIN_AXIS, Q_MAX_AXIS], zeroline=True)
    fig3.update_yaxes(title="Eₓ (N/C)", range=[-YMAX, YMAX], zeroline=True)
    style_axes_black(fig3)
    st.plotly_chart(fig3, use_container_width=True, config=PLOT_CFG_STATIC)

st.caption("🔴 O ponto vermelho indica a situação atual.")
