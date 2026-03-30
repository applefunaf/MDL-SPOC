import os
import sys
from schrodinger import structure
from schrodinger.structutils import build
from schrodinger.structutils import minimize
from schrodinger.structutils import analyze
from schrodinger.protein import assignment

def log(msg):
    print(f"[Info] {msg}")

def apply_ph_protonation(st, target_ph):
    """
    使用 PropKa 计算 pKa 并根据目标 pH 分配质子化状态。
    同时优化氢键网络（翻转 His/Asn/Gln）。
    """
    try:
        # 1. 确保有基本的氢原子
        build.add_hydrogens(st)
        
        # 3. 执行分配
        log(f"  -> 正在计算 pKa 并应用 pH={target_ph} 的质子化状态 (PropKa)...")
        
        assignment.ProtAssign(st, 
                              interactive=False,
                              do_flips=True,
                              use_propka=True,
                              propka_pH=float(target_ph))
        
        return True
    except Exception as e:
        print(f"[Error] 质子化分配失败: {e}")
        return False

def write_clean_pqr(st, filename):
    """
    提取 OPLS 电荷和半径，仅保留蛋白质原子，写入 PQR
    """
    protein_atoms = analyze.evaluate_asl(st, "protein")
    
    if not protein_atoms:
        print("[Error] 结构中未检测到蛋白质原子！")
        return

    with open(filename, 'w') as f:
        serial = 1
        for i in protein_atoms:
            atom = st.atom[i]
            # 经过 Assigner 和 Minimize 后，这里的电荷是 pH 修正后的
            charge = atom.partial_charge
            radius = atom.vdw_radius
            
            atom_name = atom.pdbname.strip()
            if len(atom_name) < 4:
                atom_name_fmt = f" {atom_name:<3}"
            else:
                atom_name_fmt = f"{atom_name:<4}"

            line = "{:6s}{:5d} {:4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:8.4f}{:8.4f}\n".format(
                "ATOM", serial % 100000, atom_name_fmt, atom.pdbres.strip(),
                atom.chain.strip(), atom.resnum,
                atom.x, atom.y, atom.z, charge, radius
            )
            f.write(line)
            serial += 1

def process_structure(input_file, output_file, ph_val=7.0):
    log(f"=== 开始处理: {input_file} (pH: {ph_val}) ===")
    
    if not os.path.exists(input_file):
        print(f"[Error] 文件不存在: {input_file}")
        return

    # 1. 读取结构
    try:
        st = structure.StructureReader.read(input_file)
    except Exception as e:
        print(f"[Error] 无法读取文件 {input_file}: {e}")
        return

    # 2. 不执行突变 (Skipped)

    # 3. 应用 pH 相关的质子化状态
    build.delete_hydrogens(st)
    
    if not apply_ph_protonation(st, ph_val):
        print("[Warning] pH 处理出现问题，尝试使用默认加氢继续...")
        build.add_hydrogens(st)

    # 4. 溶剂环境弛豫 (Energy Minimization)
    log("  -> 正在溶剂环境中优化构象 (OPLS4/VSGB)...")
    min_obj = minimize.Minimizer(struct=st)
    min_obj.minimize()
    
    # 5. 清洗并导出 PQR
    log(f"  -> 导出 PQR: {output_file}")
    write_clean_pqr(st, output_file)
    log("完成。")

if __name__ == "__main__":
    # 设置工作目录为脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    input_cif = "7p76.cif"
    output_pqr = "7p76_relaxed.pqr"
    
    process_structure(input_cif, output_pqr)
