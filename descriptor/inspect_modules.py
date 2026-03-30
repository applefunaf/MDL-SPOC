import sys
import os

print(f"Python executable: {sys.executable}")

try:
    import schrodinger
    print(f"schrodinger path: {schrodinger.__path__}")
except ImportError:
    print("schrodinger module not found")
    sys.exit(1)

def check_import(name):
    try:
        mod = __import__(name, fromlist=[''])
        print(f"[OK] {name}")
        return mod
    except ImportError as e:
        print(f"[FAIL] {name}: {e}")
        return None

# Check the problematic imports
check_import("schrodinger.structure")
check_import("schrodinger.structutils.build")
check_import("schrodinger.structutils.minimize")
check_import("schrodinger.structutils.analyze")
protein = check_import("schrodinger.protein")
if protein:
    print(f"schrodinger.protein dir: {dir(protein)}")

check_import("schrodinger.protein.mutator")
assignment = check_import("schrodinger.protein.assignment")
if assignment:
    print(f"schrodinger.protein.assignment dir: {dir(assignment)}")

# Check for potential correct locations
check_import("schrodinger.application.bioluminate")
bioluminate_protein = check_import("schrodinger.application.bioluminate.protein")
if bioluminate_protein:
    print(f"schrodinger.application.bioluminate.protein dir: {dir(bioluminate_protein)}")

check_import("schrodinger.application.bioluminate.protein")
check_import("schrodinger.application.bioluminate.mutation")
check_import("schrodinger.application.prime")

check_import("schrodinger.application.prepwizard")
prep = check_import("schrodinger.application.prepwizard")
if prep:
    print(f"schrodinger.application.prepwizard dir: {dir(prep)}")

build_mod = check_import("schrodinger.structutils.build")
if build_mod:
    print(f"mutate in build: {'mutate' in dir(build_mod)}")



