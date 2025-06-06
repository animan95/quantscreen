import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from io import StringIO, BytesIO
import base64
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pyscf import gto, scf, dft, tools
import tempfile

# Load models
drug_model = joblib.load("models/model_nn.pkl")
tox_models = {}
model_dir = "models"
for fname in os.listdir(model_dir):
    if fname.startswith("tox_model_") and fname.endswith(".pkl"):
        tox_models[fname.replace("tox_model_", "").replace(".pkl", "")] = joblib.load(os.path.join(model_dir, fname))

# Human-readable names for toxicity models
tox_descriptions = {
    "NR-AhR": "Aryl hydrocarbon receptor",
    "NR-AR": "Androgen receptor",
    "NR-AR-LBD": "Androgen receptor ligand-binding domain",
    "NR-Aromatase": "Aromatase enzyme",
    "NR-ER": "Estrogen receptor",
    "NR-ER-LBD": "Estrogen receptor ligand-binding domain",
    "NR-PPAR-gamma": "Peroxisome proliferator-activated receptor gamma",
    "SR-ARE": "Antioxidant response element",
    "SR-ATAD5": "ATAD5 (DNA damage response)",
    "SR-HSE": "Heat shock response element",
    "SR-MMP": "Mitochondrial membrane potential",
    "SR-p53": "Tumor suppressor p53"
}

# Function to compute descriptors
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(fp), mol

# Function to run PySCF single-point calculation and generate cube files
def run_pyscf_calc(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = conf.GetPositions() * 1.88973  # convert to Bohr

    mol_str = "\n".join(f"{atom} {x:.6f} {y:.6f} {z:.6f}" for atom, (x, y, z) in zip(atoms, coords))
    mol_obj = gto.M(atom=mol_str, unit='Bohr', basis='sto-3g')
    mf = dft.RKS(mol_obj)
    mf.xc = 'b3lyp'
    mf.kernel()

    homo = mf.mo_energy[mol_obj.nelectron//2 - 1]
    lumo = mf.mo_energy[mol_obj.nelectron//2]
    dip = mf.dip_moment(unit='Debye')
    energy = mf.e_tot

    with tempfile.TemporaryDirectory() as tmpdir:
        tools.cubegen.orbital(mol_obj, f"{tmpdir}/homo.cube", mf.mo_coeff[:, mol_obj.nelectron//2 - 1])
        tools.cubegen.orbital(mol_obj, f"{tmpdir}/lumo.cube", mf.mo_coeff[:, mol_obj.nelectron//2])
        with open(f"{tmpdir}/homo.cube", "r") as f:
            homo_cube = f.read()
        with open(f"{tmpdir}/lumo.cube", "r") as f:
            lumo_cube = f.read()

    return homo, lumo, energy, np.linalg.norm(dip), homo_cube, lumo_cube

# Page layout
st.title("Quantum-Inspired Drug Screening App")
st.markdown("Upload SMILES/XYZ file or paste SMILES/XYZ string to assess drug-likeness, toxicity, and QM properties.")

uploaded_file = st.file_uploader("Upload CSV file (with SMILES or XYZ coordinates)", type=["csv"])
text_input = st.text_area("Or paste a SMILES or XYZ string here")
run_qm = st.checkbox("Run PySCF QM Calculation")
show_orbitals = st.checkbox("Generate HOMO/LUMO Orbitals (requires QM calculation)", value=False, disabled=not run_qm)

input_data = []
input_mode = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'smiles' in df.columns:
        st.success("SMILES column detected. Running predictions...")
        input_data = df['smiles'].tolist()
        input_mode = "smiles"
    elif 'xyz' in df.columns:
        st.warning("XYZ support requires pre-parsed coordinate handling. Please add parser.")
        st.stop()
    else:
        st.error("No recognizable column found. Please provide a 'smiles' or 'xyz' column.")

elif text_input:
    if " " in text_input or text_input.startswith("ATOM"):
        st.warning("XYZ parsing is not yet implemented.")
        st.stop()
    else:
        st.success("Single SMILES string detected. Running prediction...")
        input_data = [text_input.strip()]
        input_mode = "smiles"

if input_data and input_mode == "smiles":
    feats_mols = [featurize(smi) for smi in input_data]
    valid_idx = [i for i, (f, m) in enumerate(feats_mols) if f is not None]

    results = []
    for i in valid_idx:
        smi = input_data[i]
        fp, mol = feats_mols[i]
        fp = fp.reshape(1, -1)
        drug_pred = drug_model.predict(fp)[0]
        tox_preds = {name: model.predict(fp)[0] for name, model in tox_models.items()}
        results.append({"SMILES": smi, "Drug-like": drug_pred, **tox_preds})

        st.markdown(f"### 🧪 Molecule Structure for: `{smi}`")
        st.image(Draw.MolToImage(mol, size=(300, 300)))

        st.markdown("### ☣️ Toxicity Alerts")
        toxic_flags = [name for name, val in tox_preds.items() if val == 1]
        if toxic_flags:
            for t in toxic_flags:
                desc = tox_descriptions.get(t, t)
                st.error(f"⚠️ Toxicity Alert: {t} — {desc}")
        else:
            st.success("No toxicity alerts detected.")

        if run_qm:
            st.markdown("### 🧮 Quantum Properties via PySCF")
            try:
                homo, lumo, energy, dipole, homo_cube, lumo_cube = run_pyscf_calc(smi)
                st.write(f"**HOMO Energy:** {homo:.6f} Hartree")
                st.write(f"**LUMO Energy:** {lumo:.6f} Hartree")
                st.write(f"**Total Energy:** {energy:.6f} Hartree")
                st.write(f"**Dipole Moment:** {dipole:.6f} Debye")

                if show_orbitals:
                    st.download_button("Download HOMO Cube", homo_cube, file_name="homo.cube")
                    st.download_button("Download LUMO Cube", lumo_cube, file_name="lumo.cube")
            except Exception as e:
                st.error(f"PySCF calculation failed: {e}")

    result_df = pd.DataFrame(results)
    st.dataframe(result_df)

    # Toxicity profile heatmap
    tox_cols = [col for col in result_df.columns if col not in ["SMILES", "Drug-like"]]

    def highlight_tox(val):
        color = "background-color: red" if val == 1 else "background-color: lightgreen"
        return color

    st.markdown("### 🔬 Toxicity Profile (Color-coded)")
    styled_df = result_df[["SMILES"] + tox_cols].style.applymap(highlight_tox, subset=tox_cols)
    st.dataframe(styled_df)

    st.markdown("### 🧯 Toxicity Heatmap (0=Safe, 1=Toxic)")
    fig, ax = plt.subplots(figsize=(10, min(0.5 * len(result_df), 10)))
    sns.heatmap(result_df[tox_cols].astype(float), cmap="Reds", cbar=True, xticklabels=tox_cols, yticklabels=False, ax=ax)
    st.pyplot(fig)

    # Download link
    csv = result_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="drug_screen_results.csv">Download Results CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

