import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from models.spatial_gcn import Spatial_GCN_Layer
from models.temporal_brain import Temporal_Brain_Layer
from utils.dataset import NTUSkeletonDataset
from utils.pipeline_config import (
    checkpoint_directory,
    dataset_num_classes,
    dataset_split_path,
    select_pipeline_input,
)

# Fixed two-stream test evaluation.
DATASET_NAME = 'xsub120'
RUN_ID = 'run17'
TEST_DIR = dataset_split_path(DATASET_NAME, 'test')
NUM_CLASSES = dataset_num_classes(DATASET_NAME)
BATCH_SIZE = 16
FUSION_WEIGHTS = {
    'JBV': 0.5,
    'B': 0.5,
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_brain(in_channels, folder_path):
    gcn = Spatial_GCN_Layer(in_channels=in_channels, out_channels=128).to(device)
    transformer = Temporal_Brain_Layer(
        embed_dim=128,
        num_heads=4,
        max_frames=100,
        max_bodies=2,
    ).to(device)
    classifier = nn.Linear(128, NUM_CLASSES).to(device)

    gcn.load_state_dict(torch.load(
        f'{folder_path}/best_gcn.pth',
        map_location=device,
        weights_only=True,
    ))
    transformer.load_state_dict(torch.load(
        f'{folder_path}/best_transformer.pth',
        map_location=device,
        weights_only=True,
    ))
    classifier.load_state_dict(torch.load(
        f'{folder_path}/best_classifier.pth',
        map_location=device,
        weights_only=True,
    ))

    gcn.eval()
    transformer.eval()
    classifier.eval()
    return gcn, transformer, classifier


def main():
    wandb.init(
        project='HAR-REAT',
        name='EVAL-2-STREAM-FUSION',
        config={
            'architecture': 'Late Fusion (9-Ch Kinematic + 3-Ch Structural)',
            'dataset': 'NTU-RGB+D 120 X-Subject Test Set',
            'batch_size': BATCH_SIZE,
            'fusion_weights': FUSION_WEIGHTS,
        },
    )

    print(f'Device in-use: {device.type.upper()}')
    print('Loading test data...')
    test_dataset = NTUSkeletonDataset(
        data_folder=TEST_DIR,
        max_frames=100,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    print('Loading two-stream architecture...')
    experts = {
        'JBV': load_brain(
            9,
            checkpoint_directory(DATASET_NAME, RUN_ID, 'jbv'),
        ),
        'B': load_brain(
            3,
            checkpoint_directory(DATASET_NAME, RUN_ID, 'bones'),
        ),
    }

    total_samples = 0
    correct_predictions = 0

    print('Running fixed two-stream ensemble evaluation...')
    loop = tqdm(
        test_dataloader,
        total=len(test_dataloader),
        leave=True,
        desc='Evaluating',
    )

    with torch.no_grad():
        for batched_data, body_mask, labels in loop:
            batched_data = batched_data.to(device)
            body_mask = body_mask.to(device)
            labels = labels.to(device).long().view(-1)

            batch_size, bodies, frames, joints, channels = batched_data.shape
            stream_inputs = {
                'JBV': select_pipeline_input(batched_data, 'jbv'),
                'B': select_pipeline_input(batched_data, 'bones'),
            }
            fused_probs = torch.zeros(
                batch_size,
                NUM_CLASSES,
                device=device,
            )

            for stream_name, stream_input in stream_inputs.items():
                gcn, transformer, classifier = experts[stream_name]
                input_data = stream_input.reshape(
                    batch_size,
                    bodies,
                    frames,
                    joints,
                    stream_input.shape[-1],
                )
                input_data = input_data.reshape(
                    batch_size * bodies,
                    frames,
                    joints,
                    stream_input.shape[-1],
                )

                features = gcn(input_data)
                feature_frames = features.shape[1]
                transformer_input = torch.cat(
                    [
                        features,
                        transformer.global_node.expand(
                            batch_size * bodies,
                            feature_frames,
                            1,
                            128,
                        ),
                    ],
                    dim=2,
                )
                video_representation = transformer(
                    transformer_input,
                    batch_size,
                    bodies,
                    body_mask=body_mask,
                )
                probabilities = torch.softmax(
                    classifier(video_representation),
                    dim=1,
                )
                fused_probs += FUSION_WEIGHTS[stream_name] * probabilities

            predicted_classes = torch.argmax(fused_probs, dim=1)
            correct_predictions += torch.eq(
                predicted_classes,
                labels,
            ).sum().item()
            total_samples += labels.size(0)

            current_accuracy = (correct_predictions / total_samples) * 100
            loop.set_postfix(acc=f'{current_accuracy:.2f}%')
            wandb.log({'Running Accuracy (%)': current_accuracy})

    accuracy = (correct_predictions / total_samples) * 100
    wandb.log({
        'Final Ensemble Accuracy (%)': accuracy,
        'Total Unseen Videos': total_samples,
    })
    wandb.finish()

    print('\n' + '=' * 50)
    print('2-STREAM ENSEMBLE EVALUATION COMPLETE')
    print(f'Total Unseen Videos Processed: {total_samples}')
    print(f'Final Top-1 Accuracy:          {accuracy:.2f}%')
    print('=' * 50)


if __name__ == '__main__':
    main()
