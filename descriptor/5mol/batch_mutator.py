import csv
import os
import sys
import argparse
from schrodinger import structure
from schrodinger.structutils import build
from schrodinger.structutils import minimize
from schrodinger.structutils import analyze
# 引入蛋白质质子化分配模块
from schrodinger.protein import assignment
from schrodinger.application.bioluminate import protein

def log(msg):
    print(f"[Info] {msg}")

def apply_ph_protonation(st, target_ph):
    """
    使用 PropKa 计算 pKa 并根据目标 pH 分配质子化状态。
    同时优化氢键网络（翻转 His/Asn/Gln）。
    """
    try:
        # 1. 确保有基本的氢原子 (Assigner 需要完整的原子结构)
        build.add_hydrogens(st)
        
        # 3. 执行分配
        # 这是一个计算密集型步骤，可能会花费几秒到几十秒
        log(f"  -> 正在计算 pKa 并应用 pH={target_ph} 的质子化状态 (PropKa)...")
        
        # 使用 ProtAssign class
        assignment.ProtAssign(st, 
                              interactive=False,
                              do_flips=True,
                              use_propka=True,
                              propka_pH=float(target_ph))
        
        return True
    except Exception as e:
        print(f"[Error] 质子化分配失败: {e}")
        return False

def process_single_structure(input_file, output_file, mutations_str, ph_val):
    log(f"=== 开始处理: {input_file} (pH: {ph_val}) ===")
    
    # 1. 读取结构
    try:
        st = structure.StructureReader.read(input_file)
    except Exception as e:
        print(f"[Error] 无法读取文件 {input_file}: {e}")
        return False

    # 2. 执行突变
    if mutations_str and mutations_str.lower() != 'none':
        mut_list = mutations_str.split(';')
        mutations_to_apply = []

        for mut in mut_list:
            mut = mut.strip()
            if not mut: continue
            try:
                parts = mut.split(':')
                if len(parts) == 3:
                    chain_id, resnum, new_res = parts
                    resnum = int(resnum)
                    
                    # Mutator 需要 1 字母代码
                    if len(new_res) == 3:
                        new_res_1 = structure.RESIDUE_MAP_3_TO_1_LETTER.get(new_res.upper())
                        if new_res_1:
                            new_res = new_res_1
                    
                    mutations_to_apply.append((chain_id, resnum, new_res))
                else:
                     print(f"[Warning] 突变参数错误 '{mut}' (需 Chain:ResNum:Type)，跳过。")
            except ValueError as ve:
                print(f"[Warning] 突变格式处理错误 '{mut}': {ve}，跳过。")
        
        if mutations_to_apply:
            log(f"  -> 执行多位点并发突变: {mutations_to_apply}")
            try:
                # 使用 Bioluminate Mutator 进行并发突变
                mutator_obj = protein.Mutator(st, mutations_to_apply)
                results = list(mutator_obj.generate())
                
                if results:
                    # 取第一个生成的结构（通常只会生成一个结构，包含所有指定的点突变）
                    st = results[0].struct
                    log(f"  -> 突变完成 (Bioluminate Mutator)")
                else:
                    log("  -> [Warning] Mutator 未生成任何结构")
            except Exception as e:
                print(f"[Error] Mutator 执行失败: {e}")

    # 3. 应用 pH 相关的质子化状态
    # 按照最佳实践：先删除所有氢原子，再重新添加和分配
    build.delete_hydrogens(st)
    
    if not apply_ph_protonation(st, ph_val):
        print("[Warning] pH 处理出现问题，尝试使用默认加氢继续...")
        build.add_hydrogens(st)

    # 4. 溶剂环境弛豫 (Energy Minimization)
    log("  -> 正在溶剂环境中优化构象 (OPLS4/VSGB)...")
    min_obj = minimize.Minimizer(struct=st)
    min_obj.minimize()
    # Minimizer 可能会直接修改传入的 structural object，或者是 update()，这里确保使用最新的
    st_optimized = st

    # 5. 清洗并导出 PQR
    log(f"  -> 导出 PQR: {output_file}")
    write_clean_pqr(st_optimized, output_file)
    
    return True

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

def main():
    parser = argparse.ArgumentParser(description="Schrodinger 批量突变/pH处理/PQR生成工具")
    parser.add_argument("csv_file", help="CSV 文件: Input, Output, Mutations, pH")
    args = parser.parse_args()

    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    csv_path = args.csv_file
    if not os.path.exists(csv_path):
        # 尝试在脚本目录下寻找
        candidate = os.path.join(script_dir, csv_path)
        if os.path.exists(candidate):
            csv_path = candidate
        else:
            print(f"错误: 找不到文件 {args.csv_file}")
            sys.exit(1)

    print("========================================")
    print("开始批量处理 (含 pH 计算)...")
    print("========================================")

    success_count = 0
    total_count = 0

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'): continue
            
            total_count += 1
            input_f = row[0].strip()
            
            # 如果输入文件不存在，尝试在脚本目录下寻找
            if not os.path.exists(input_f):
                candidate_input = os.path.join(script_dir, input_f)
                if os.path.exists(candidate_input):
                    input_f = candidate_input
                else:
                    # 尝试在脚本目录下进行不区分大小写的查找
                    input_basename = os.path.basename(input_f)
                    try:
                        for f_name in os.listdir(script_dir):
                            if f_name.lower() == input_basename.lower():
                                candidate_input = os.path.join(script_dir, f_name)
                                if os.path.exists(candidate_input):
                                    input_f = candidate_input
                                    print(f"[Warning] 自动修正文件名大小写: {row[0]} -> {f_name}")
                                    break
                    except OSError:
                        pass

            output_f = row[1].strip()
            
            # 解析突变
            mutations = row[2].strip() if len(row) > 2 else None
            
            # 解析 pH (默认为 7.0)
            try:
                ph_val = float(row[3].strip()) if len(row) > 3 and row[3].strip() else 7.0
            except ValueError:
                print(f"[Warning] pH 值 '{row[3]}' 无效，使用默认值 7.0")
                ph_val = 7.0
            
            if process_single_structure(input_f, output_f, mutations, ph_val):
                success_count += 1
            print("-" * 40)

    print(f"\n全部完成. 成功: {success_count}/{total_count}")

if __name__ == "__main__":
    main()