import json
import copy


class BaseConfig:
    """
    Base config.
    """

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = cls()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `Config` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_parsed_args(cls, args):
        config = cls()
        for attr in config.__dict__:
            config.__setattr__(attr, args.__getattribute__(attr))
        return config

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class QCModelConfig(BaseConfig):
    """Configuration class to store the configuration of our models.
    """
    def __init__(self,
                 nb_measurements=4,
                 hidden_dim=16,
                 nb_encoder_layers=1,
                 num_attention_heads=4,
                 intermediate_dim=32,
                 hidden_act="gelu",
                 positional_encoding_dropout=0.1,
                 max_position_embeddings=100,
                 hidden_dropout_prob=0.1,
                 attention_dropout_prob=0.1,
                 initializer_range=0.02,
                 layer_norm_type="annotated",
                 layer_norm_eps=1e-12):
        """
        Constructs QC_Transformer Configuration object
        :param nb_measurements: Number of measurements
        :param hidden_dim: Size of the encoder and decoder layers
        :param nb_encoder_layers: Number of encoder layers
        :param num_attention_heads: Number of attention heads
        :param intermediate_dim: Size of the feedforward layer
        :param hidden_act: The nonlinear activation to be used
        :param positional_encoding_dropout: Positional encoding dropout rate
        :param max_position_embeddings: Maximun sequence length that this model may ever used. This is typically set to
            something large (512, 1024, 2048)
        :param hidden_dropout_prob: Dropout rate for all fully connected layers
        :param attention_dropout_prob: Dropout rate for attention sublayers
        :param initializer_range: The stddev of the truncated normal initializer for initializing the weight matrices
        :param layer_norm_type: The type of LayerNorm to be used
        :param layer_norm_eps: The epsilon use for layer norm
        """
        # TODO: ADD SUPPORT FOR DIFFERENT LAYER NORMS

        self.nb_measurements = nb_measurements
        self.hidden_dim = hidden_dim
        self.nb_encoder_layers = nb_encoder_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_dim = intermediate_dim
        self.hidden_act = hidden_act
        self.positional_encoding_dropout = positional_encoding_dropout
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_type = layer_norm_type
        self.layer_norm_eps=layer_norm_eps


class TrainConfig(BaseConfig):

    def __init__(self,
                 # nb_epochs=20,
                 nb_samples=-1,
                 exponent_index=-1,
                 eval_nb_samples=100_000,
                 sampling_batch_size=100_000,
                 mini_batch_size=100_000,
                 accumulation_step=1,
                 # batch_size=10_000,
                 max_step=5000,
                 lr=0.01,
                 beta=10.0,
                 tau=0.1,
                 device="cpu",
                 save_dir="",
                 evaluate=True,
                 final_state="Graph",
                 writer="",
                 exp_nb=0,
                 data_random_seed=13,
                 model_random_seed=-1,
                 # circuit config
                 nb_qbits=-1,
                 povm="4Pauli",
                 circuit_file="",
                 initial_product_state="",
                 circuit_type="basic",
                 # qc model config
                 nb_measurements=4,
                 hidden_dim=16,
                 nb_encoder_layers=1,
                 num_attention_heads=4,
                 intermediate_dim=32,
                 hidden_act="gelu",
                 positional_encoding_dropout=0.1,
                 max_position_embeddings=100,
                 hidden_dropout_prob=0.1,
                 attention_dropout_prob=0.1,
                 initializer_range=0.02,
                 layer_norm_type="annotated",
                 layer_norm_eps=1e-12,
                 ):
        """
        Construct training configuration object
        # :param nb_epochs: The number of epochs that each gate will be trained.
        :param nb_samples: The number of samples used in each epoch.
        :param eval_nb_samples: Number of samples used for evaluation.
        :param sampling_batch_size: Batch size for sample generation
        :param mini_batch_size: Mini batch size for each training step
        :param accumulation_step: Number of steps to accumulate gradients
        # :param batch_size: Batch size for training
        :param max_step: Number of steps in training
        :param beta: total time for imaginary/real time evolution
        :param tau: unit time step for imaginary/real time evolution
        :param device: cpu or cuda
        :param save_dir: directory where to save the trained model
        """

        # self.nb_epochs = nb_epochs
        self.nb_samples = nb_samples
        self.eval_nb_samples = eval_nb_samples
        self.sampling_batch_size = sampling_batch_size
        self.mini_batch_size = mini_batch_size
        self.accumulation_step = accumulation_step
        # self.batch_size = batch_size
        self.max_step = max_step
        self.lr = lr
        self.beta = beta
        self.tau = tau
        self.device = device
        self.save_dir = save_dir
        self.final_state = final_state
        self.evaluate = evaluate
        self.exponent_index = exponent_index
        self.writer = writer
        self.data_random_seed = data_random_seed
        self.model_random_seed = model_random_seed
        # circuit config
        self.nb_qbits = nb_qbits
        self.povm = povm
        self.circuit_file = circuit_file
        self.initial_product_state = initial_product_state
        self.circuit_type = circuit_type
        # qc model config
        self.nb_measurements = nb_measurements
        self.hidden_dim = hidden_dim
        self.nb_encoder_layers = nb_encoder_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_dim = intermediate_dim
        self.hidden_act = hidden_act
        self.positional_encoding_dropout = positional_encoding_dropout
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_type = layer_norm_type
        self.layer_norm_eps=layer_norm_eps


class CircuitConfig(BaseConfig):

    def __init__(self,
                 nb_qbits=-1,
                 povm="4Pauli",
                 circuit_file="",
                 initial_product_state="",
                 circuit_type="basic",
                 ):
        self.nb_qbits = nb_qbits
        self.povm = povm
        self.circuit_file = circuit_file
        self.initial_product_state = initial_product_state
        self.circuit_type = circuit_type

