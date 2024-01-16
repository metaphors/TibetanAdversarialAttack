import OpenAttack
from datasets import load_dataset


def dataset_mapping(data):
    return {
        "x": data["text"],
        "y": data["label"],
    }


def main():
    print("New Attacker")
    # TSAttacker
    # attacker = OpenAttack.attackers.PWWSAttacker(lang="tibetan")
    # TSTricker (syllable level)
    # attacker = OpenAttack.attackers.PWWSAttacker2(lang="tibetan")
    # TSTricker (word level)
    attacker = OpenAttack.attackers.PWWSAttacker3(lang="tibetan")

    print("Building model")
    clsf = OpenAttack.loadVictim("BERT.TIBETAN-BERT_TNCC-DOCUMENT")

    print("Loading dataset")
    dataset = load_dataset("dataset_loader/tncc-document.py", split="test").map(function=dataset_mapping)

    print("Start attack")
    attack_eval = OpenAttack.AttackEval(attacker, clsf, "tibetan", metrics=[
        OpenAttack.metric.EditDistance()
    ])
    attack_eval.eval(dataset, visualize=True, progress_bar=True)


if __name__ == "__main__":
    main()
