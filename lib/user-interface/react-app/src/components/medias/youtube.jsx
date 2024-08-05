
/*
https://youtu.be/AVaFEanJAUs?si=cWXjA7MeVzOZnWbX
https://www.youtube.com/watch?v=AVaFEanJAUs&t=400s
https://img.youtube.com/vi/AVaFEanJAUs/hqdefault.jpg
*/

import React from 'react';
import { useEffect, useState } from "react";
import OneVideo from './video'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faAngleRight, faAngleDown } from '@fortawesome/free-solid-svg-icons'

function getYoutubeVideoId(url) {
    let s = ''
    let sa = url.split('youtu.be/')
    if (sa.length > 1) {
        s = sa[1]
        sa = s.split('?')
        s = sa[0]
        return s
    }
    sa = url.split('youtube.com/watch?v=')
    if (sa.length > 1) {
        s = sa[1]
        sa = s.split('&')
        s = sa[0]
        return s
    }
    return null
}

export default function youtubeVideos({ urls }) {

    const [show, setShow] = useState(false)
    const onClick = () => setShow(!show)

    const valid_urls = urls.filter(u => getYoutubeVideoId(u))

    return (
        <div
            style={{
                margin: "0.5rem",
                padding: "0.5rem",
                borderTop: "solid 1pt",
                borderColor: "#D3D3D3"
            }}
        >
            <div style={{
                display: "flex",
                flexDirection: "row",
                justifyContent: "space-between",
            }}>
                <FontAwesomeIcon icon={show ? faAngleDown : faAngleRight}
                    onClick={onClick} />
                <div style={{
                    marginLeft: "auto",
                    marginRight: "auto",
                    fontWeight: "600"
                }}>Videos</div>
            </div>
            {
                show &&
                <div style={{
                    display: "flex",
                    flexDirection: "row",
                    justifyContent: "left",
                    alignItems: "center",
                    gap: "0.5rem",
                    flexWrap: "wrap",
                    marginTop: "1rem"
                }}>
                    {
                        valid_urls.map((u, i) =>
                            <OneVideo key={i} url={u} id={getYoutubeVideoId(u)} />
                        )
                    }
                </div>
            }
        </div>
    );
}

/*
const baseClass = 'rich-text-video';

type Source = 'youtube'

const sourceLabels: Record<Source, string> = {
    youtube: 'YouTube',
};

const Element = props => {
    const { attributes, children, element } = props;
    const { source, id } = element;
    const selected = useSelected();
    const focused = useFocused();
    const [title, setTitle] = useState('');

    useEffect(() => {
        const fetchData = async () => {
            if (source !== 'youtube') {
                setTitle(`${sourceLabels[source]} Video: ${id}`);
                return;
            }
            const data = await fetch(
                `https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=${id}&format=json`,
            );
            const json = await data.json();
            setTitle(json.title)
        };
        fetchData()
    }, [id, title]);

    return (
        <div
            className={[baseClass, selected && focused && `${baseClass}--selected`]
                .filter(Boolean)
                .join(' ')}
            contentEditable={false}
            {...attributes}
        >
            {source === 'youtube' && (<img
                src={`https://img.youtube.com/vi/${id}/hqdefault.jpg`}
                style={{ maxWidth: '100%' }}
            />)}
            <div className={`${baseClass}__wrap`}>
                <div className={`${baseClass}__label`}>
                    <VideoIcon />
                    <div className={`${baseClass}__title`}>
                        {title}
                    </div>
                </div>
            </div>
            {children}
        </div>
    );
};
*/
