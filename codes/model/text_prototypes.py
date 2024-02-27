import torch
import torch.nn as nn
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import pickle
from torch.nn import DataParallel
import warnings
from codes.conversation import conv_templates, SeparatorStyle
from tqdm import tqdm
from codes.model.MetaLlama import DyGLlamaForCausalLM
import os
import pickle

warnings.filterwarnings("ignore")
device = torch.device("cpu")


class TransposedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(TransposedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x.t()).t()


# 构建 Multi-Head Cross-Attention
class CrossAttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionModel, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# edge_representation = torch.randn(10, 512)  
#
# # Text Prototypes
# text_prototypes = [
#     "Node", "Vertex", "Edge", "Link", "Degree", "Path", "Neighbor", "Community", "Subgraph",
#     "Grow", "Shrink", "Connect", "Disconnect", "Strengthen", "Weaken", "Activate", "Deactivate",
#     "Timestamp", "Time", "Duration", "Interval", "Sequence", "Periodic", "Transient",
#     "Flow", "Stream", "Source", "Sink", "Influence", "Affect", "Propagate", "Spread",
#     "Merge", "Split", "Centrality", "Cluster",
#     "Increase", "Decrease", "Expand", "Contract", "Rise", "Fall", "Up", "Down", "Escalate", "Diminish",
#     "Emerge", "Evolve", "Transition", "Morph", "Fluctuate", "Oscillate", "Vary", "Modify",
#     "Transform", "Mutate", "Revolutionize", "Develop", "Advance", "Regress", "Degrade",
#     "Upgrade", "Update", "Integrate", "Disintegrate", "Aggregate", "Disperse", "Coalesce",
#     "Fission", "Fusion", "Synthesize", "Decompose", "Reform", "Reconfigure", "Restructure",
#     "Reinforce", "Undermine", "Optimize", "Deteriorate", "Enhance", "Impair", "Proliferate",
#     "Reciprocate", "Interact", "Interlock", "Intersect", "Parallel", "Diverge", "Converge",
#     "Align", "Misalign", "Balance", "Imbalance", "Synchronize", "Asynchronize", "Coordinate",
#     "Discoordinate", "Complement", "Supplement", "Superimpose", "Overlay", "Underlay",
#     "Encapsulate", "Enclose", "Envelop", "Expose", "Conceal", "Reveal", "Manifest", "Obscure",
#     "Illuminate", "Shade", "Highlight", "Shadow", "Echo", "Silence", "Resonate", "Attenuate",
#     "Amplify", "Muffle", "Distort", "Clarify", "Blur", "Focus", "Broaden", "Narrow",
#     "Deepen", "Shallow", "Elevate", "Depress", "Accelerate", "Decelerate", "Mobilize",
#     "Stagnate", "Catalyze", "Inhibit", "Prompt", "Delay", "Hasten", "Retard", "Expedite",
#     "Pace", "Rhythm", "Cycle", "Wave", "Pulse", "Beat", "Cadence", "Tempo", "Speed",
#     "Momentum", "Inertia", "Force", "Pressure", "Tension", "Relaxation", "Strain", "Stress",
#     "Load", "Burden", "Capacity", "Volume", "Magnitude", "Quantity", "Measure", "Metric",
#     "Scale", "Proportion", "Ratio", "Fraction", "Percentage", "Quantity", "Amount", "Total",
#     "Sum", "Difference", "Product", "Quotient", "Derivative", "Integral", "Variable", "Constant",
#     "Parameter", "Indicator", "Criterion", "Benchmark", "Standard", "Norm", "Model", "Pattern",
#     "Template", "Framework", "Structure", "Architecture", "Design", "Blueprint", "Prototype",
#     "Sample", "Example", "Instance", "Case", "Scenario", "Situation", "Context", "Environment"
# ]
# text_prototypes += [
#     "Elevate", "Lower", "Enlarge", "Minimize", "Accelerate", "Slow", "Intensify", "Alleviate",
#     "Maximize", "Reduce", "Multiply", "Divide", "Augment", "Diminish", "Exaggerate", "Understate",
#     "Boost", "Suppress", "Heighten", "Lessen", "Broaden", "Restrict", "Magnify", "Minify",
#     "Amplify", "Deafen", "Brighten", "Darken", "Strengthen", "Weaken", "Harden", "Soften",
#     "Solidify", "Liquefy", "Thicken", "Thin", "Widen", "Tighten", "Expand", "Contract",
#     "Swollen", "Shrunken", "Elongate", "Shorten", "Enlarge", "Narrow", "Advance", "Retreat",
#     "Ascend", "Descend", "Rise", "Fall", "Climb", "Sink", "Soar", "Plummet", "Surge", "Recede",
#     "Proliferate", "Dwindle", "Blossom", "Wither", "Flourish", "Perish", "Burgeon", "Fade",
#     "Bloom", "Decay", "Sprout", "Rot", "Grow", "Shrink", "Evolve", "Devolve", "Progress", "Regress",
#     "Thrive", "Fail", "Succeed", "Falter", "Win", "Lose", "Gain", "Forfeit", "Capture", "Release",
#     "Grab", "Let go", "Seize", "Relinquish", "Acquire", "Dispose", "Collect", "Disperse",
#     "Gather", "Scatter", "Assemble", "Disassemble", "Build", "Dismantle", "Create", "Destroy",
#     "Generate", "Eradicate", "Spawn", "Extinguish", "Breed", "Sterilize", "Produce", "Annihilate",
#     "Construct", "Demolish", "Fabricate", "Obliterate", "Forge", "Disintegrate", "Synthesize", "Dissolve",
#     "Invent", "Extinct", "Develop", "Degrade", "Formulate", "Corrode", "Compose", "Decompose",
#     "Aspire", "Despair", "Hope", "Dread", "Expect", "Fear", "Anticipate", "Overlook",
#     "Foresee", "Neglect", "Predict", "Misjudge", "Project", "Disregard", "Envision", "Ignore",
#     "Dream", "Forget", "Fantasize", "Dismiss", "Scheme", "Overestimate", "Plan", "Underestimate",
#     "Contemplate", "Overlook", "Meditate", "Miss", "Ponder", "Disregard", "Reflect", "Neglect",
#     "Consider", "Oversight", "Deliberate", "Unaware", "Think", "Unconscious", "Ruminate", "Unnoticed",
#     "Muse", "Undetected", "Speculate", "Invisible", "Conjecture", "Hidden", "Theorize", "Obscure",
#     "Hypothesize", "Camouflaged", "Infer", "Concealed", "Deduce", "Shrouded", "Surmise", "Veiled",
#     "Guess", "Disguised", "Postulate", "Masked", "Suppose", "Covered", "Presume", "Cloaked",
#     "Assume", "Screened", "Believe", "Shielded", "Conclude", "Protected", "Decide", "Safeguarded",
#     "Judge", "Secured", "Resolve", "Insulated", "Determine", "Guarded", "Settle", "Fortified",
#     "Choose", "Defended", "Select", "Barricaded", "Prefer", "Walled", "Opt", "Fenced",
#     "Decree", "Enclosed", "Command", "Caged", "Dictate", "Penned", "Order", "Confined",
#     "Instruct", "Imprisoned", "Direct", "Ensnared", "Mandate", "Trapped", "Prescribe", "Captured",
#     "Advise", "Sequestered", "Recommend", "Isolated", "Suggest", "Detached", "Propose", "Aloof",
#     "Urge", "Apart", "Counsel", "Separated", "Advocate", "Divided", "Champion", "Parted",
#     "Support", "Sundered", "Back", "Cleft", "Endorse", "Rent", "Promote", "Torn",
#     "Encourage", "Ripped", "Inspire", "Shattered", "Motivate", "Fractured", "Stimulate", "Splintered",
#     "Provoke", "Cracked", "Instigate", "Broken", "Incite", "Damaged", "Trigger", "Harmed",
#     "Spark", "Injured", "Ignite", "Wounded", "Fuel", "Bruised", "Foster", "Scarred",
#     "Nurture", "Maimed", "Cultivate", "Lacerated", "Breed", "Gashed", "Raise", "Slashed",
#     "Rear", "Stabbed", "Grow", "Pierced", "Develop", "Penetrated", "Train", "Perforated",
#     "Educate", "Transpierced", "Instruct", "Impaled", "Teach", "Skewered", "Guide", "Speared",
#     "Coach", "Pricked", "Mentor", "Stung", "Tutor", "Bitten", "Influence", "Nipped",
#     "Affect", "Pecked", "Impact", "Gnawed", "Shape", "Chewed", "Mold", "Munched",
#     "Form", "Gobbled", "Craft", "Devoured", "Design", "Ingested", "Engineer", "Swallowed",
#     "Create", "Consumed", "Make", "Eaten", "Produce", "Absorbed", "Construct", "Digested",
#     "Build", "Metabolized", "Fabricate", "Assimilated", "Assemble", "Integrated", "Erect", "Incorporated",
#     "Establish", "Engrafted", "Set up", "Implanted", "Organize", "Injected", "Arrange", "Infused",
#     "Coordinate", "Infiltrated", "Manage", "Permeated", "Administer", "Saturated", "Oversee", "Soaked",
#     "Supervise", "Drenched", "Control", "Steeped", "Govern", "Imbued", "Rule", "Impregnated",
#     "Direct", "Filled", "Lead", "Loaded", "Guide", "Laden", "Steer", "Packed",
#     "Pilot", "Stuffed", "Navigate", "Crammed", "Drive", "Brims", "Operate", "Bursting",
#     "Conduct", "Overflowing", "Handle", "Flooded", "Wield", "Engulfed", "Use", "Submerged",
#     "Utilize", "Immersed", "Employ", "Drowned", "Apply", "Swamped", "Practice", "Sunk",
#     "Exercise", "Plunged", "Perform", "Dipped", "Carry out", "Sloshed", "Execute", "Doused",
#     "Implement", "Sprinkled", "Enact", "Sprayed", "Carry through", "Splashed", "Fulfill", "Spattered",
#     "Complete", "Dripped", "Achieve", "Drizzled", "Accomplish", "Poured", "Realize", "Discharged",
#     "Attain", "Emitted", "Reach", "Exuded", "Grasp", "Seeped", "Seize", "Oozed",
#     "Capture", "Leaked", "Obtain", "Trickled", "Acquire", "Dribbled", "Gain", "Drooled",
#     "Secure", "Salivated", "Fetch", "Sweated", "Snatch", "Excreted", "Grab", "Ejected",
#     "Take", "Expelled", "Collect", "Released", "Gather", "Squirted", "Amass", "Spurted",
#     "Accumulate", "Jetted", "Stockpile", "Gushed", "Hoard", "Spewed", "Store", "Erupted",
#     "Keep", "Exploded", "Hold", "Blasted", "Possess", "Fired", "Own", "Shot",
#     "Have", "Aimed"]
