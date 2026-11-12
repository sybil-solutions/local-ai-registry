import json
import unittest

import catalog
import lab

R = {"card": "some-card", "model": "some-model"}
PREFER = {"some-card/some-model": {"recipe": "some-model.vllm.512k.4x.mine"}}


class OwnerPickTest(unittest.TestCase):
    def test_a_decision_names_one_recipe_and_only_a_lab_proven_one_is_put_first(self):
        self.assertTrue(catalog.owner_pick(PREFER, R, "some-model.vllm.512k.4x.mine", 0))
        self.assertFalse(catalog.owner_pick(PREFER, R, "some-model.vllm.512k.4x.mine", 1))  # legacy
        self.assertFalse(catalog.owner_pick(PREFER, R, "some-model.vllm.512k.4x.mine", 2))  # reported
        self.assertFalse(catalog.owner_pick(PREFER, R, "some-model.vllm.512k.4x", 0))  # another recipe of the model
        self.assertFalse(catalog.owner_pick({}, R, "some-model.vllm.512k.4x.mine", 0))
        self.assertFalse(catalog.owner_pick({"some-card/some-model": {"engine": "sglang"}}, R, "some-model.vllm.512k.4x.mine", 0))

    def test_every_recipe_an_owner_decision_names_leads_its_setup_once_lab_proven(self):
        cat = json.loads(catalog.build())
        prefer = json.loads((lab.REG / "prefer.json").read_text(encoding="utf-8"))
        for key, decision in prefer.items():
            if not decision.get("recipe"):
                continue
            card = key.split("/")[0]
            setups = [s for s in cat["cards"][card]["setups"] if any(k.endswith("/" + decision["recipe"]) for k in s["picks"] + s["more"])]
            self.assertEqual(len(setups), 1, f"{key}: the recipe the decision names is in the catalog once")
            full = next(k for k in setups[0]["picks"] + setups[0]["more"] if k.endswith("/" + decision["recipe"]))
            proof = cat["recipes"][full]["proof"][0]
            if not proof.get("reported") and not proof.get("legacy"):
                self.assertEqual(setups[0]["picks"][0], full, f"{key}: an accepted owner pick is the recommended recipe")


if __name__ == "__main__":
    unittest.main()
