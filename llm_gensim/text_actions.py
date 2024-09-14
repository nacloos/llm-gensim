from pathlib import Path
import yaml

from llm_gensim.generate_sim import sample_hexaco_personality
from llm_gensim.separate_questions import make_sim_pipeline, eval_pipeline
from llm_gensim.generate_sim import simulate


save_dir = Path(__file__).parent / "results" / Path(__file__).stem


# TODO: with activities as sentences, used regex to answer questions
# but can't decode personalities
if __name__ == "__main__":

    model_id = "claude-3-5-sonnet-20240620"
    pipeline_name = "personality_activities"
    batch_name = "batch1"
    
    pipe_dir = save_dir / pipeline_name / model_id / batch_name / "gen"
    sim_dir = save_dir / pipeline_name / model_id / batch_name / "sim"
    pipeline_path = Path(__file__).parent / "configs" / "text_actions" / (pipeline_name + ".yaml")
    pipeline_config = yaml.safe_load(pipeline_path.read_text())

    pipeline = make_sim_pipeline(model_id, pipeline_config, pipe_dir)

    num_steps = 500
    num_samples = 1
    num_runs = 5
    for factor_name in ["Extraversion", "Conscientiousness", "Openness to Experience", "Agreeableness", "Honesty-Humility"]:
        _save_dir = sim_dir / factor_name / f"num_steps{num_steps}"
        eval_pipeline(pipeline, factor_name=factor_name, num_sim_steps=num_steps, num_samples=num_samples, num_runs=num_runs, save_dir=_save_dir)
    