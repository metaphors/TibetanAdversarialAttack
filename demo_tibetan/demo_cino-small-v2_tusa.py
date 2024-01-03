import OpenAttack
from datasets import load_dataset


def dataset_mapping(data):
    return {
        "x": data["text"],
        "y": data["label"],
    }


def main():
    print("New Attacker")
    attacker = OpenAttack.attackers.PWWSAttacker(lang="tibetan")

    print("Building model")
    clsf = OpenAttack.loadVictim("XLMROBERTA.CINO-SMALL-V2_TUSA")

    print("Loading dataset")
    dataset = load_dataset("dataset_loader/tusa.py", split="test").map(function=dataset_mapping)

    print("Start attack")
    attack_eval = OpenAttack.AttackEval(attacker, clsf, "tibetan", metrics=[
        OpenAttack.metric.EditDistance()
    ])
    attack_eval.eval(dataset, visualize=True, progress_bar=True)


if __name__ == "__main__":
    main()
