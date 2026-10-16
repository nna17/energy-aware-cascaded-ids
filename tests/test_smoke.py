"""End-to-end on synthetic data: teacher ensemble -> supernet KD -> sub-model eval -> trees."""
import torch

from eids.eval.evaluator import evaluate, evaluate_subnet, metrics_from_predictions
from eids.models.baselines.cnn1d_diab import CNN1D
from eids.models.baselines.trees import tree_grid
from eids.models.elastic import ElasticMLP, SearchSpace
from eids.models.teacher import TeacherEnsemble, build_teacher
from eids.training.classifier import train_classifier
from eids.training.supernet import train_supernet
from eids.utils.config import Config, load_config

CPU = torch.device("cpu")


def test_end_to_end(synthetic_dm):
    dm = synthetic_dm
    t_cfg = load_config("configs/train/teacher.yaml", epochs=2, patience=None)
    members = []
    for path in ("configs/model/teacher_mlp.yaml", "configs/model/teacher_ftt.yaml"):
        m = build_teacher(load_config(path), dm.n_features, dm.n_classes)
        train_classifier(m, dm, t_cfg, CPU, seed=0)
        members.append(m)
    teacher = TeacherEnsemble(members)
    assert evaluate(teacher, dm, "test", CPU)["macro_f1"] > 0.3

    model_cfg = load_config("configs/model/elastic_mlp.yaml")
    s_cfg = load_config("configs/train/supernet.yaml", epochs=2, patience=None, bn_recalib_batches=5)
    space = SearchSpace.from_config(model_cfg, dm.n_features)
    net = ElasticMLP(dm.n_features, dm.n_classes, model_cfg.max_widths)
    train_supernet(net, space, dm, s_cfg, CPU, seed=0, teacher=teacher)
    for sub in (space.largest(), space.smallest()):
        m = evaluate_subnet(net, sub, dm, "test", CPU, recalib_batches=5)
        assert 0.0 <= m["minority_recall"] <= 1.0

    cnn = CNN1D(dm.n_features, dm.n_classes, [8, 16])
    train_classifier(cnn, dm, Config(epochs=1, batch_size=512, lr=1e-3), CPU, seed=0)

    for tree in tree_grid(["lightgbm", "xgboost", "random_forest"], [5], [8]):
        tree.fit(dm.X["train"], dm.y["train"])
        assert tree.size_kb() > 0
        assert 0 < tree.n_comparisons() <= 5 * dm.n_classes * 8  # <= trees x max depth (8 leaves -> depth <= 7)
        assert metrics_from_predictions(dm, "test", tree.predict(dm.X["test"]))["macro_f1"] > 0.2
