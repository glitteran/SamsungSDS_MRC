#-*- coding: utf-8 -*-
from transformers import Trainer

class QuestionAnsweringTrainer(Trainer):
    """
    model, args, train_dataset, eval_dataset, tokenizer, data_collator, compute_metrics가 추가로 들어감. 
    post_process_function: formatted_predictions(guid, prediction_text), references(guit, answers) 생성. metric을 계산하기 위한 이전 과정 
    compute metrics: post_process_function의 리턴값인 EvalPrediction을 인풋으로 받아서  exact, f1값 리턴. 
    """
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        ## evaluation_loop() 함수의 작동을 확인하려면 https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.get_train_dataloader
        ## return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally: #try 실행 후 항상 실행되는 것. 
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            ## metrics.keys() 
            ##  "exact": 100.0 * sum(exact_raw.values()) / total,
            ##  "f1": 100.0 * sum(f1_raw.values()) / total,
            ##  "total": total
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    value = metrics.pop(key)
                    metrics[f"{metric_key_prefix}_{key}"] =value

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    """
    def train(
        self,
        ##python main.py --resume 270851bert-base/checkpoint-500/
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
    
    train 함수에서 resume_from_checkpoint 인자로 할 수 있는 것 
    1. resume_from_checkpoint(str ckpt path)안에 파일 접근하기
    2. config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME)) #CONFIG_NAME = "config.json"
    3. state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu") #WEIGHTS_NAME = "pytorch_model.bin"
    4. self._load_optimizer_and_scheduler(resume_from_checkpoint)
    5. self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, "trainer_state.json"))
    6. self._load_rng_state(resume_from_checkpoint) #rng_state.pth 
    """