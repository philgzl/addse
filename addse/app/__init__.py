import typer

from .eval import app as eval_app
from .ldopt import app as ldopt_app
from .train import app as train_app

app = typer.Typer(pretty_exceptions_show_locals=False)
app.add_typer(eval_app)
app.add_typer(ldopt_app)
app.add_typer(train_app)
