using UnityEngine;

public class BlazeTimelineController : MonoBehaviour
{
    public BlazeDataExtraction blaze;
    public bool playOnEnable = true;
    public bool stopOnDisable = true;

    private void Awake()
    {
        if (blaze == null) blaze = GetComponent<BlazeDataExtraction>();
    }

    private void OnEnable()
    {
        if (!playOnEnable) return;
        if (blaze == null) return;
        blaze.PlayFromStart();
    }

    private void OnDisable()
    {
        if (!stopOnDisable) return;
        if (blaze == null) return;
        blaze.StopPlayback();
    }
}
