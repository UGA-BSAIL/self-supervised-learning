# BodyBatchUpdateMetadataVideosMetadataBatchUpdatePatch


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**metadata** | [**UavVideoMetadata**](UavVideoMetadata.md) |  |
**videos** | [**List[ObjectRef]**](ObjectRef.md) |  |

## Example

```python
from mallard_client.models.body_batch_update_metadata_videos_metadata_batch_update_patch import BodyBatchUpdateMetadataVideosMetadataBatchUpdatePatch

# TODO update the JSON string below
json = "{}"
# create an instance of BodyBatchUpdateMetadataVideosMetadataBatchUpdatePatch from a JSON string
body_batch_update_metadata_videos_metadata_batch_update_patch_instance = BodyBatchUpdateMetadataVideosMetadataBatchUpdatePatch.from_json(json)
# print the JSON string representation of the object
print BodyBatchUpdateMetadataVideosMetadataBatchUpdatePatch.to_json()

# convert the object into a dict
body_batch_update_metadata_videos_metadata_batch_update_patch_dict = body_batch_update_metadata_videos_metadata_batch_update_patch_instance.to_dict()
# create an instance of BodyBatchUpdateMetadataVideosMetadataBatchUpdatePatch from a dict
body_batch_update_metadata_videos_metadata_batch_update_patch_form_dict = body_batch_update_metadata_videos_metadata_batch_update_patch.from_dict(body_batch_update_metadata_videos_metadata_batch_update_patch_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)
